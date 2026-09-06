import urllib.request
import json

def run_tests():
    # 1. Test split=test
    test_puzzles = json.loads(urllib.request.urlopen('http://localhost:3000/api/puzzles?split=test&limit=5').read().decode('utf-8'))
    print("Test split count:", len(test_puzzles))
    first_test_id = test_puzzles[0]['id']
    print("First test puzzle:", first_test_id, test_puzzles[0]['name'])

    # 2. Test split=novelty
    novel_puzzles = json.loads(urllib.request.urlopen('http://localhost:3000/api/puzzles?split=novelty&limit=5').read().decode('utf-8'))
    print("Novelty split count:", len(novel_puzzles))
    first_novel_id = novel_puzzles[0]['id']
    print("First novel puzzle:", first_novel_id, novel_puzzles[0]['name'])

    # 3. Context model across demo counts (1, 3, 5)
    for k in [1, 3, 5]:
        url = f"http://localhost:3000/api/predict?puzzle_id={first_test_id}&model_type=context&demo_count={k}"
        ctx = json.loads(urllib.request.urlopen(url).read().decode('utf-8'))
        print(f"Context Model (demo={k}): source={ctx.get('source')}, exact_match={ctx.get('exact_match')}, latency={ctx.get('latency_ms')}ms, confidence={ctx.get('confidence')}")

    # 4. Optimization model dynamic inference
    url = f"http://localhost:3000/api/predict?puzzle_id={first_test_id}&model_type=optimization&demo_count=3"
    opt = json.loads(urllib.request.urlopen(url).read().decode('utf-8'))
    print(f"Optimization Model (demo=3): source={opt.get('source')}, exact_match={opt.get('exact_match')}, latency={opt.get('latency_ms')}ms, loss_steps={len(opt.get('loss_curve', []))}")

    # 5. Context model on novelty puzzle (recolor)
    url = f"http://localhost:3000/api/predict?puzzle_id={first_novel_id}&model_type=context&demo_count=5&novelty=novel"
    ctx_novel = json.loads(urllib.request.urlopen(url).read().decode('utf-8'))
    print(f"Novelty Context Model: source={ctx_novel.get('source')}, exact_match={ctx_novel.get('exact_match')}, latency={ctx_novel.get('latency_ms')}ms")

if __name__ == '__main__':
    run_tests()
