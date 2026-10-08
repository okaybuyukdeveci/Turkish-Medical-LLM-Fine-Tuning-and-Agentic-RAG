"""Run the four Qwen conditions using the unchanged LangGraph workflow."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from scripts.llm.qwen import QwenChat, load_qwen

MODES = ('base', 'sft', 'base_agentic', 'sft_agentic')
ANSWER_FORMAT = 'İlk satıra yalnızca "Doğru cevap: X" yaz (X: A, B, C, D veya E). Sonrasında açıklama yazabilirsin.'


def read_jsonl(path):
    """Read UTF-8 JSON records, skipping blank lines and preserving file order."""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    """Validate benchmark data and run the requested base/SFT graph conditions.

    Reads settings from YAML, resolves data paths relative to that file, and
    checks question/answer structure before loading model weights. With
    --validate-only, exits after those checks. With --limit, evaluates a prefix
    of the validated dataset for a small trial.

    Each question starts with fresh conversation state. Predictions, timing,
    retrieval counts, and errors are flushed to per-mode JSONL files. Failed or
    unparseable answers count as incorrect in the final accuracy summary.
    """
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('benchmark.yaml'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--modes', nargs='+', choices=MODES)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    def resolve(value):
        """Resolve a configured filesystem path relative to the YAML directory."""
        return (path.parent / value).resolve()
    benchmark = config['benchmark']
    questions = read_jsonl(resolve(benchmark['questions']))
    keys = read_jsonl(resolve(benchmark['answer_key']))
    answers = {row['id']: row['correct_option'] for row in keys}
    ids = [row['id'] for row in questions]
    if len(set(ids)) != len(ids) or len(answers) != len(keys) or set(ids) != set(answers):
        raise ValueError('Question and answer IDs must be unique and match exactly')
    if len(ids) != benchmark['expected_questions']:
        raise ValueError('Unexpected question count; use the reviewed final dataset')
    for row in questions:
        if [option['label'] for option in row['options']] != list('ABCDE') or any(not option['text'].strip() for option in row['options']):
            raise ValueError(f"Invalid options: {row['id']}")
        if not row['question'].strip() or answers[row['id']] not in 'ABCDE':
            raise ValueError(f"Invalid question or answer: {row['id']}")
        if re.search(r'cevap\s*(?:anahtarı|anahtari|:)', row['question'], re.I):
            raise ValueError(f"Answer leakage: {row['id']}")
    print(f'Validated {len(ids)} questions', flush=True)
    if args.validate_only:
        return
    if args.limit is not None:
        if args.limit <= 0:
            parser.error('--limit must be positive')
        questions = questions[:args.limit]
    modes = args.modes or benchmark['modes']
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError('Invalid modes')
    config['model']['adapter'] = str(resolve(config['model']['adapter']))
    shared = load_qwen(config['model'], config['generation'], with_adapter=any('sft' in mode for mode in modes))
    retriever = None
    if any(mode.endswith('_agentic') for mode in modes):
        from scripts.retrieval.retriever import load_retriever
        from scripts.settings import RagSettings
        retriever = load_retriever(RagSettings.from_env())
    output = resolve(benchmark['output'])
    output.mkdir(parents=True, exist_ok=True)
    run = output / time.strftime('%Y%m%d-%H%M%S')
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {}
    for mode in modes:
        llm = QwenChat(shared.model, shared.tokenizer, adapter='sft' in mode, **config['generation'])
        graph = None
        if mode.endswith('_agentic'):
            from scripts.workflow.graph import build_graph
            graph = build_graph(llm, llm, retriever)
        records = []
        with (run / f'{mode}.jsonl').open('w', encoding='utf-8') as handle:
            for row in questions:
                prompt = row['question'] + '\n\n' + '\n'.join(f"{o['label']}) {o['text']}" for o in row['options']) + '\n\n' + ANSWER_FORMAT
                start = time.monotonic()
                result = {'id': row['id'], 'mode': mode, 'expected': answers[row['id']]}
                try:
                    messages = graph.invoke({'messages': [HumanMessage(content=prompt)]})['messages'] if graph else [llm.invoke([SystemMessage(content=ANSWER_FORMAT), HumanMessage(content=prompt)])]
                    response = messages[-1].content
                    match = re.match(r'\s*Doğru cevap:\s*([A-E])\b', response)
                    prediction = match.group(1) if match else None
                    calls = [call for message in messages for call in getattr(message, 'tool_calls', [])]
                    evidence = []
                    for message in messages:
                        artifact = getattr(message, 'artifact', None)
                        if isinstance(artifact, dict):
                            evidence.append({
                                'status': artifact.get('status'),
                                'sources': artifact.get('sources', []),
                                'content': message.content,
                            })
                    evidence_supplied = any(
                        item['status'] == 'ok' and item['sources'] and str(item['content']).strip()
                        for item in evidence
                    )
                    result.update(response=response, prediction=prediction, correct=prediction == answers[row['id']],
                        retrieval_queries=[call['args'].get('query') for call in calls],
                        retrieval_evidence=evidence,
                        evidence_supplied_to_answer=evidence_supplied,
                        retrieval_calls=sum(len(getattr(m, 'tool_calls', [])) for m in messages),
                        retrieval_errors=sum(getattr(m, 'artifact', {}).get('status') == 'error' for m in messages if isinstance(getattr(m, 'artifact', None), dict)))
                except Exception as exc:
                    result.update(prediction=None, correct=False, error=f'{type(exc).__name__}: {exc}')
                result['seconds'] = time.monotonic() - start
                records.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                print(mode, row['id'], result['correct'],
                    f"retrieval_calls={result.get('retrieval_calls', 'unknown')}",
                    f"evidence_supplied={result.get('evidence_supplied_to_answer', False)}",
                    f"queries={json.dumps(result.get('retrieval_queries', []), ensure_ascii=False)}",
                    flush=True)
                for item in result.get('retrieval_evidence', []):
                    print('RETRIEVAL', row['id'], f"status={item['status']}",
                        f"sources={json.dumps(item['sources'], ensure_ascii=False)}", flush=True)
        summary[mode] = {'questions': len(records), 'accuracy': sum(r['correct'] for r in records) / len(records), 'errors': sum('error' in r for r in records)}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(run, flush=True)


if __name__ == '__main__':
    main()
