# 🧠 WordNet Visualizer

An interactive, graph-based visualization of [Princeton WordNet](https://wordnet.princeton.edu/) — explore semantic relations between English words in real time.

> **[🚀 Try it live →](https://wordnet-vis.onrender.com/)**

---

## ✨ Features

- **Interactive Graph** — Explore words as an expanding network. Click to inspect, double-click to expand, drag to rearrange.
- **Pure WordNet Data** — All relations come directly from NLTK WordNet, not statistical approximations.
- **Rich Relations** — Synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms — all color-coded.
- **Per-Synset Details** — Click any word to see each sense (synset) with its own definition, POS tag, and specific relations. No more mixing word senses.
- **Shortest Path Finder** — Select two words (Ctrl/Cmd + click) to find the shortest semantic path through WordNet's taxonomy tree.
- **Cumulative Search** — Search multiple words without clearing previous results.
- **Node Deletion** — Select nodes and press Delete/Backspace to remove them (cascades to children).
- **Reset** — One click to start fresh.

## 📸 How It Works

| Action | Result |
|--------|--------|
| **Search** a word | Adds it to the graph with all its WordNet relations |
| **Double-click** a node | Expands that word's relations |
| **Click** a node | Shows per-synset definitions and relations in the info panel |
| **Ctrl+Click** two nodes | Finds and visualizes the shortest path between them |
| **Delete/Backspace** | Removes selected nodes and their subtrees |

### Example: Path from "tiger" to "deer"

```
tiger → tiger cub → cub → young mammal → fawn → deer
```

Each step is a real hypernym/hyponym link in WordNet's taxonomy.

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML + CSS + JavaScript, [vis-network](https://visjs.github.io/vis-network/docs/network/) |
| Backend | Python [Flask](https://flask.palletsprojects.com/) + [NLTK WordNet](https://www.nltk.org/howto/wordnet.html) |
| Deployment | [Render.com](https://render.com) |

## 🚀 Run Locally

```bash
git clone https://github.com/XindaLi304/wordnet-vis.git
cd wordnet-vis

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the server
python3 server.py
# Open http://localhost:5001
```

## 📂 Project Structure

```
├── index.html          # Main page
├── style.css           # Dark theme + glassmorphism styles
├── app.js              # Frontend: graph logic, interactions, API calls
├── server.py           # Flask backend: WordNet queries + pathfinding
├── requirements.txt    # Python dependencies
├── Procfile            # Render.com start command
└── build.sh            # Render.com build script
```

## 🎨 Relation Color Guide

| Color | Relation | Meaning |
|-------|----------|---------|
| 🟣 Purple | Root | Searched word |
| 🟢 Green | Synonym | Same meaning |
| 🔴 Red | Antonym | Opposite meaning |
| 🔵 Blue | Hypernym | Broader category ("is a kind of") |
| 🩵 Cyan | Hyponym | More specific ("a type of") |
| 🩷 Pink | Meronym | Part of |
| 💜 Violet | Holonym | Contains / whole of |

## 📄 License

MIT

---

Built with ❤️ using Princeton WordNet and NLTK.
