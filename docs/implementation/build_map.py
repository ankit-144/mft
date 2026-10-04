"""Embed a curated responsibility map with source links pinned to the current commit."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, check=True).stdout.decode().strip()


def main():
    data = json.loads((HERE / 'map-content.json').read_text())
    nodes = {n['id']: n for n in data['nodes']}
    assert len(nodes) == len(data['nodes']), 'Duplicate node IDs'
    tracked = set(git('ls-tree', '-r', '--name-only', 'HEAD').splitlines())
    for n in nodes.values():
        if n['parent']:
            assert n['parent'] in nodes, n['id']
        for child in n.get('children', []):
            assert child in nodes and nodes[child]['parent'] == n['id'], (n['id'], child)
        for f in n.get('files', []):
            assert f['path'] in tracked, (n['id'], f['path'])
        for child in n.get('sequence', []):
            assert child in nodes, (n['id'], child)
        for item in n.get('jumps', []):
            assert item['id'] in nodes, (n['id'], item)
        for item in n.get('connections', []):
            assert item['target'] in nodes, (n['id'], item)
    data['revision'] = git('rev-parse', 'HEAD')
    data['source_url'] = 'https://github.com/ankit-144/mft/blob/' + data['revision'] + '/'
    (HERE / 'map-data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    html = (HERE / 'map-template.html').read_text().replace('<!-- MAP_DATA -->', '<script type="application/json" id="map-data">' + payload + '</script>')
    (HERE / 'guide.html').write_text(html)
    print(json.dumps({'nodes':len(nodes), 'source_files':len({f['path'] for n in nodes.values() for f in n.get('files',[])}), 'html_bytes':len(html.encode()), 'revision':data['revision'], 'output':str(HERE/'guide.html')}))


if __name__ == '__main__':
    main()
