"""
WordNet Visualization Backend
Flask server using NLTK WordNet for authentic semantic relations and pathfinding.
"""

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import nltk
import os

# Download WordNet data if not present
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import wordnet as wn

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lemma_names(synsets):
    """Extract unique lemma names from a list of synsets."""
    names = set()
    for s in synsets:
        for lemma in s.lemmas():
            name = lemma.name().replace('_', ' ')
            names.add(name)
    return list(names)


def get_all_synsets(word):
    """Get all synsets for a word."""
    return wn.synsets(word.replace(' ', '_'))


POS_MAP = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective satellite', 'r': 'adverb'}


# ---------------------------------------------------------------------------
# API: Get related words
# ---------------------------------------------------------------------------

@app.route('/api/related/<word>')
def get_related(word):
    synsets = get_all_synsets(word)

    if not synsets:
        return jsonify({
            'word': word,
            'found': False,
            'synsets': [],
            'synonyms': [], 'antonyms': [],
            'hypernyms': [], 'hyponyms': [],
            'meronyms': [], 'holonyms': []
        })

    # Build per-synset grouped data
    synset_list = []
    # Also collect merged flat sets for graph expansion
    all_synonyms = set()
    all_antonyms = set()
    all_hypernyms = set()
    all_hyponyms = set()
    all_meronyms = set()
    all_holonyms = set()

    for s in synsets:
        # Synonyms for this synset
        syns = set()
        ants = set()
        for lemma in s.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                syns.add(name)
            for ant in lemma.antonyms():
                ants.add(ant.name().replace('_', ' '))

        # Hypernyms
        hypers = set()
        for h in s.hypernyms():
            for lemma in h.lemmas():
                hypers.add(lemma.name().replace('_', ' '))

        # Hyponyms
        hypos = set()
        for h in s.hyponyms():
            for lemma in h.lemmas():
                hypos.add(lemma.name().replace('_', ' '))

        # Meronyms
        meros = set()
        for m in s.part_meronyms() + s.substance_meronyms() + s.member_meronyms():
            for lemma in m.lemmas():
                meros.add(lemma.name().replace('_', ' '))

        # Holonyms
        holos = set()
        for h in s.part_holonyms() + s.substance_holonyms() + s.member_holonyms():
            for lemma in h.lemmas():
                holos.add(lemma.name().replace('_', ' '))

        synset_list.append({
            'synset_id': s.name(),
            'pos': POS_MAP.get(s.pos(), s.pos()),
            'definition': s.definition(),
            'examples': s.examples(),
            'synonyms': sorted(syns),
            'antonyms': sorted(ants),
            'hypernyms': sorted(hypers),
            'hyponyms': sorted(hypos),
            'meronyms': sorted(meros),
            'holonyms': sorted(holos)
        })

        all_synonyms.update(syns)
        all_antonyms.update(ants)
        all_hypernyms.update(hypers)
        all_hyponyms.update(hypos)
        all_meronyms.update(meros)
        all_holonyms.update(holos)

    return jsonify({
        'word': word,
        'found': True,
        'synsets': synset_list,
        # Flat merged lists for graph expansion
        'synonyms': sorted(all_synonyms)[:15],
        'antonyms': sorted(all_antonyms)[:10],
        'hypernyms': sorted(all_hypernyms)[:10],
        'hyponyms': sorted(all_hyponyms)[:15],
        'meronyms': sorted(all_meronyms)[:10],
        'holonyms': sorted(all_holonyms)[:10]
    })


# ---------------------------------------------------------------------------
# API: Shortest path between two words
# ---------------------------------------------------------------------------

@app.route('/api/path/<word1>/<word2>')
def get_path(word1, word2):
    synsets1 = get_all_synsets(word1)
    synsets2 = get_all_synsets(word2)

    if not synsets1 or not synsets2:
        return jsonify({'found': False, 'error': f'Word not found in WordNet', 'path': []})

    # Find the pair of synsets with the shortest path
    best_path = None
    best_dist = float('inf')
    best_s1 = None
    best_s2 = None

    for s1 in synsets1:
        for s2 in synsets2:
            try:
                dist = s1.shortest_path_distance(s2)
                if dist is not None and dist < best_dist:
                    best_dist = dist
                    best_s1 = s1
                    best_s2 = s2
            except Exception:
                continue

    if best_s1 is None or best_s2 is None:
        return jsonify({'found': False, 'error': 'No path exists between these words in WordNet', 'path': []})

    # Reconstruct the actual path via hypernym tree
    path_synsets = best_s1.hypernym_paths()
    path2_synsets = best_s2.hypernym_paths()

    # Find LCA (Lowest Common Ancestor) to reconstruct path
    path_words = reconstruct_path(best_s1, best_s2)

    return jsonify({
        'found': True,
        'distance': best_dist,
        'synset1': best_s1.name(),
        'synset2': best_s2.name(),
        'path': path_words
    })


def reconstruct_path(s1, s2):
    """Reconstruct the shortest path between two synsets via BFS, tracking edge direction."""
    from collections import deque

    # BFS from s1, tracking parents and edge direction
    # parent[synset] = (parent_synset, edge_type)
    # edge_type: 'hypernym' means we went UP from child to parent
    #            'hyponym' means we went DOWN from parent to child
    parent = {s1: (None, None)}
    queue = deque([s1])
    found = False

    while queue and not found:
        current = queue.popleft()
        # Expand hypernyms (going UP)
        for nb in current.hypernyms():
            if nb not in parent:
                parent[nb] = (current, 'hypernym')  # current → nb is going UP
                if nb == s2:
                    found = True
                    break
                queue.append(nb)
        if found:
            break
        # Expand hyponyms (going DOWN)
        for nb in current.hyponyms():
            if nb not in parent:
                parent[nb] = (current, 'hyponym')  # current → nb is going DOWN
                if nb == s2:
                    found = True
                    break
                queue.append(nb)

    if not found:
        w1 = s1.lemmas()[0].name().replace('_', ' ')
        w2 = s2.lemmas()[0].name().replace('_', ' ')
        return [{'word': w1, 'synset': s1.name(), 'definition': s1.definition(), 'edge_to_next': 'hypernym'},
                {'word': w2, 'synset': s2.name(), 'definition': s2.definition(), 'edge_to_next': None}]

    # Reconstruct path (reversed)
    path = []
    cur = s2
    while cur is not None:
        parent_synset, edge_type = parent[cur]
        lemma = cur.lemmas()[0].name().replace('_', ' ')
        path.append({
            'word': lemma,
            'synset': cur.name(),
            'definition': cur.definition(),
            'edge_from_prev': edge_type  # how we got TO this node
        })
        cur = parent_synset

    path.reverse()

    # Convert edge_from_prev to edge_to_next for frontend convenience
    # edge_to_next tells the frontend: what relation does the edge FROM this node TO the next node represent?
    for i in range(len(path) - 1):
        next_edge = path[i + 1].get('edge_from_prev')
        # If BFS went UP (hypernym) from path[i] to path[i+1], then path[i] "is a kind of" path[i+1]
        # If BFS went DOWN (hyponym) from path[i] to path[i+1], then path[i+1] "is a kind of" path[i]
        path[i]['edge_to_next'] = next_edge
    path[-1]['edge_to_next'] = None

    # Remove internal field
    for p in path:
        p.pop('edge_from_prev', None)

    return path


# ---------------------------------------------------------------------------
# Serve static files
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    print("🧠 WordNet Visualization Server starting...")
    print("   Open http://localhost:5001 in your browser")
    app.run(debug=True, port=5001)
