/**
 * WordNet Visualizer App
 * Uses local Flask + NLTK WordNet backend for authentic semantic relations.
 */

// Use relative URL since Flask serves both the frontend and API from the same origin
const API_BASE = '';

// Global State
let network = null;
let nodes = new vis.DataSet();
let edges = new vis.DataSet();
let loadedWords = new Set(); // Keep track of words we've already fetched
let wordAttributes = new Map(); // Store dictionary definitions and metadata

// UI Elements
const searchBtn = document.getElementById('searchBtn');
const resetBtn = document.getElementById('resetBtn');
const searchInput = document.getElementById('searchInput');
const loadingIndicator = document.getElementById('loading');
const infoPanel = document.getElementById('infoPanel');
const infoTitle = document.getElementById('infoTitle');
const infoBadges = document.getElementById('infoBadges');

// Styling constants
const COLORS = {
  root: '#8b5cf6', // Purple
  synonym: '#10b981', // Emerald
  antonym: '#ef4444', // Red
  hypernym: '#3b82f6', // Blue (General/Parent)
  hyponym: '#06b6d4', // Cyan (Specific/Child)
  trigger: '#f59e0b', // Amber
  meronym: '#ec4899', // Pink (Part-of)
  holonym: '#a855f7', // Purple (Whole-of)
  text: '#ffffff',
  border: 'rgba(255,255,255,0.2)',
  accent: '#3b82f6'
};

// Initialize the Network
function initNetwork() {
  const container = document.getElementById('network-container');
  const data = { nodes, edges };
  const options = {
    nodes: {
      shape: 'box',
      margin: 12,
      font: {
        color: COLORS.text,
        face: 'Inter',
        size: 16,
        bold: { size: 18 }
      },
      borderWidth: 1,
      shadow: {
        enabled: true,
        color: 'rgba(0,0,0,0.5)',
        size: 10,
        x: 0,
        y: 4
      }
    },
    edges: {
      width: 1.5,
      color: { color: 'rgba(255,255,255,0.15)', highlight: COLORS.accent },
      smooth: { type: 'continuous' },
      arrows: {
        to: { enabled: true, scaleFactor: 0.5 }
      },
      font: {
        size: 11,
        color: 'rgba(255,255,255,0.7)',
        strokeWidth: 2,
        strokeColor: '#0f172a', /* adds a dark background stroke to make it easy to read over lines */
        face: 'Inter',
        align: 'middle'
      }
    },
    physics: {
      forceAtlas2Based: {
        gravitationalConstant: -100,
        centralGravity: 0.005,
        springLength: 150,
        springConstant: 0.08
      },
      maxVelocity: 50,
      solver: 'forceAtlas2Based',
      timestep: 0.35,
      stabilization: { iterations: 150 }
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
      zoomView: true,
      dragView: true,
      multiselect: true // Allow selecting two nodes by holding Ctrl/Cmd
    }
  };

  network = new vis.Network(container, data, options);

  // Event Listeners for Network
  network.on('click', handleNodeClick);
  network.on('doubleClick', handleNodeDoubleClick);
  network.on('dragStart', () => infoPanel.classList.add('hidden'));
}

// API Interaction — Local WordNet Backend
async function fetchRelatedWords(word) {
  loadingIndicator.classList.remove('hidden');
  try {
    const res = await fetch(`${API_BASE}/api/related/${encodeURIComponent(word)}`);
    const data = await res.json();

    if (!data.found) {
      return { synonyms: [], antonyms: [], hypernyms: [], hyponyms: [], meronyms: [], holonyms: [] };
    }

    // Store definitions for info panel
    wordAttributes.set(word, data);

    return {
      synonyms: data.synonyms || [],
      antonyms: data.antonyms || [],
      hypernyms: data.hypernyms || [],
      hyponyms: data.hyponyms || [],
      meronyms: data.meronyms || [],
      holonyms: data.holonyms || []
    };
  } catch (error) {
    console.error("Error fetching from WordNet backend:", error);
    return { synonyms: [], antonyms: [], hypernyms: [], hyponyms: [], meronyms: [], holonyms: [] };
  } finally {
    loadingIndicator.classList.add('hidden');
  }
}

// Graph Manipulation
function createNodeStyle(type) {
  let background, border;
  if (type === 'root') { background = COLORS.root; border = '#a78bfa'; }
  else if (type === 'synonym') { background = COLORS.synonym; border = '#34d399'; }
  else if (type === 'antonym') { background = COLORS.antonym; border = '#f87171'; }
  else if (type === 'hypernym') { background = COLORS.hypernym; border = '#60a5fa'; }
  else if (type === 'hyponym') { background = COLORS.hyponym; border = '#22d3ee'; }
  else if (type === 'trigger') { background = COLORS.trigger; border = '#fbbf24'; }
  else if (type === 'meronym') { background = COLORS.meronym; border = '#f472b6'; }
  else if (type === 'holonym') { background = COLORS.holonym; border = '#c084fc'; }

  return {
    color: {
      background,
      border,
      highlight: { background, border: '#ffffff' },
      hover: { background, border: '#ffffff' }
    }
  };
}

function addWordToGraph(sourceWord, targetWord, relationType) {
  // If target node doesn't exist, create it
  if (!nodes.get(targetWord)) {
    nodes.add({
      id: targetWord,
      label: targetWord,
      group: relationType,
      ...createNodeStyle(relationType),
      font: { multi: 'html', bold: relationType === 'root' }
    });
  }

  // If edge doesn't exist, create it
  const edgeId = `${sourceWord}-${targetWord}`;
  const reverseEdgeId = `${targetWord}-${sourceWord}`;

  if (!edges.get(edgeId) && !edges.get(reverseEdgeId) && sourceWord !== targetWord) {
    let edgeLabel = relationType;
    let edgeColor = 'rgba(255,255,255,0.4)';
    if (relationType === 'synonym') { edgeColor = 'rgba(16, 185, 129, 0.4)'; }
    if (relationType === 'antonym') { edgeColor = 'rgba(239, 68, 68, 0.4)'; }
    if (relationType === 'hypernym') { edgeLabel = 'kind of'; edgeColor = 'rgba(59, 130, 246, 0.4)'; }
    if (relationType === 'hyponym') { edgeLabel = 'specific'; edgeColor = 'rgba(6, 182, 212, 0.4)'; }
    if (relationType === 'trigger') { edgeLabel = 'related'; edgeColor = 'rgba(245, 158, 11, 0.4)'; }

    edges.add({
      id: edgeId,
      from: sourceWord,
      to: targetWord,
      label: edgeLabel,
      color: { color: edgeColor },
      dashes: relationType === 'trigger' // dashed line for weak relationship
    });
  }
}

async function expandWord(word, isRoot = false) {
  if (loadedWords.has(word)) return;
  loadedWords.add(word);

  if (isRoot) {
    // Add root node if not existing (don't clear graph anymore)
    if (!nodes.get(word)) {
      nodes.add({
        id: word,
        label: `<b>${word}</b>`, // make root bold
        group: 'root',
        ...createNodeStyle('root')
      });
    }
  }

  const { synonyms, antonyms, hypernyms, hyponyms, meronyms, holonyms } = await fetchRelatedWords(word);

  // Add rich WordNet relations to graph
  synonyms.forEach(syn => addWordToGraph(word, syn, 'synonym'));
  antonyms.forEach(ant => addWordToGraph(word, ant, 'antonym'));
  hypernyms.forEach(gen => addWordToGraph(word, gen, 'hypernym'));
  hyponyms.forEach(spc => addWordToGraph(word, spc, 'hyponym'));
  meronyms.forEach(mer => addWordToGraph(word, mer, 'meronym'));
  holonyms.forEach(hol => addWordToGraph(word, hol, 'holonym'));
}

// Interaction Handlers
function handleNodeClick(params) {
  if (params.nodes.length === 2) {
    const word1 = params.nodes[0];
    const word2 = params.nodes[1];
    infoPanel.classList.add('hidden');
    findConnection(word1, word2);
    return;
  }

  if (params.nodes.length === 1) {
    const nodeId = params.nodes[0];
    const node = nodes.get(nodeId);
    showInfoPanel(nodeId, node.group);
  } else {
    infoPanel.classList.add('hidden');
  }
}

function handleNodeDoubleClick(params) {
  if (params.nodes.length > 0) {
    const nodeId = params.nodes[0];
    expandWord(nodeId);
  }
}

function showInfoPanel(word, type) {
  infoTitle.textContent = word;

  // Format badge
  let badgeHtml = '';
  if (type === 'root') badgeHtml = `<span class="badge root"><i class="fa-solid fa-star"></i> Root Word</span>`;
  else if (type === 'synonym') badgeHtml = `<span class="badge synonym"><i class="fa-solid fa-equals"></i> Synonym</span>`;
  else if (type === 'antonym') badgeHtml = `<span class="badge antonym"><i class="fa-solid fa-not-equal"></i> Antonym</span>`;
  else if (type === 'hypernym') badgeHtml = `<span class="badge hypernym"><i class="fa-solid fa-arrow-up"></i> Hypernym (Broader)</span>`;
  else if (type === 'hyponym') badgeHtml = `<span class="badge hyponym"><i class="fa-solid fa-arrow-down"></i> Hyponym (Narrower)</span>`;
  else if (type === 'trigger') badgeHtml = `<span class="badge trigger"><i class="fa-solid fa-link"></i> Associated</span>`;
  else if (type === 'meronym') badgeHtml = `<span class="badge" style="background:#ec4899"><i class="fa-solid fa-puzzle-piece"></i> Meronym (Part-of)</span>`;
  else if (type === 'holonym') badgeHtml = `<span class="badge" style="background:#a855f7"><i class="fa-solid fa-cubes"></i> Holonym (Whole-of)</span>`;

  // Add expansion status
  if (loadedWords.has(word)) {
    badgeHtml += `<span class="badge" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2)"><i class="fa-solid fa-check"></i> Expanded</span>`;
  }
  infoBadges.innerHTML = badgeHtml;

  // Render per-synset grouped data
  const attrs = wordAttributes.get(word);
  let existingAttrDiv = document.getElementById('wordAttrsPane');
  if (existingAttrDiv) existingAttrDiv.remove();

  if (attrs && attrs.synsets && attrs.synsets.length > 0) {
    const pane = document.createElement('div');
    pane.id = 'wordAttrsPane';
    pane.className = 'word-attributes';

    attrs.synsets.forEach((synset, idx) => {
      const block = document.createElement('div');
      block.style.cssText = 'margin-bottom: 10px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 3px solid rgba(139,92,246,0.6);';

      // Sense header: POS + definition
      const header = document.createElement('div');
      header.style.cssText = 'margin-bottom: 6px;';
      header.innerHTML = `<span class="pos-tag">[${synset.pos}]</span> <span style="opacity:0.5;font-size:11px">${synset.synset_id}</span><br/><span style="color:rgba(255,255,255,0.85)">${synset.definition}</span>`;
      block.appendChild(header);

      // Relations for this synset
      const relTypes = [
        { key: 'synonyms', label: 'Synonyms', color: '#10b981' },
        { key: 'antonyms', label: 'Antonyms', color: '#ef4444' },
        { key: 'hypernyms', label: 'Hypernyms ↑', color: '#3b82f6' },
        { key: 'hyponyms', label: 'Hyponyms ↓', color: '#06b6d4' },
        { key: 'meronyms', label: 'Meronyms ◇', color: '#ec4899' },
        { key: 'holonyms', label: 'Holonyms ◆', color: '#a855f7' }
      ];

      relTypes.forEach(rt => {
        if (synset[rt.key] && synset[rt.key].length > 0) {
          const relEl = document.createElement('div');
          relEl.style.cssText = 'margin-top: 4px; font-size: 12px;';
          const tags = synset[rt.key].map(w => `<span style="background:${rt.color}22; color:${rt.color}; padding:1px 6px; border-radius:4px; margin:1px; display:inline-block; font-size:11px">${w}</span>`).join(' ');
          relEl.innerHTML = `<span style="color:${rt.color}; font-weight:600; font-size:11px">${rt.label}:</span> ${tags}`;
          block.appendChild(relEl);
        }
      });

      pane.appendChild(block);
    });

    // Insert pane before instruction
    const instructionEl = document.querySelector('.instruction');
    infoPanel.insertBefore(pane, instructionEl);
  }

  infoPanel.classList.remove('hidden');
}

// Path Finding via WordNet Backend (server-side BFS through hypernym tree)
async function findConnection(w1, w2) {
  loadingIndicator.innerHTML = '<i class="fa-solid fa-magic"></i> Finding WordNet path...';
  loadingIndicator.classList.remove('hidden');

  try {
    const res = await fetch(`${API_BASE}/api/path/${encodeURIComponent(w1)}/${encodeURIComponent(w2)}`);
    const data = await res.json();

    if (!data.found || !data.path || data.path.length < 2) {
      alert(`No path found between "${w1}" and "${w2}" in WordNet.${data.error ? ' ' + data.error : ''}`);
      return;
    }

    // Extract word names from the path
    const pathWords = data.path.map(p => p.word);

    // Add all path nodes and edges with correct direction labels
    for (let i = 0; i < data.path.length - 1; i++) {
      const edgeType = data.path[i].edge_to_next || 'hypernym';
      addWordToGraph(data.path[i].word, data.path[i + 1].word, edgeType);
    }

    // Mark endpoints as root-styled
    if (nodes.get(pathWords[0])) {
      nodes.update({ id: pathWords[0], ...createNodeStyle('root') });
    }
    if (nodes.get(pathWords[pathWords.length - 1])) {
      nodes.update({ id: pathWords[pathWords.length - 1], ...createNodeStyle('root') });
    }

    highlightPath(pathWords);
    console.log(`WordNet path (distance ${data.distance}):`, pathWords.join(' → '));
  } catch (e) {
    console.error('Pathfinding error:', e);
    alert('Error connecting to WordNet backend. Make sure server.py is running on port 5000.');
  } finally {
    loadingIndicator.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Fetching semantics...';
    loadingIndicator.classList.add('hidden');
    network.unselectAll();
  }
}

// Highlight pathway visually
function highlightPath(pathWords) {
  const originalNodes = [];
  const originalEdges = [];

  pathWords.forEach(w => {
    let n = nodes.get(w);
    if (n) {
      originalNodes.push({ ...n });
      nodes.update({ id: w, borderWidth: 4, color: { border: '#eab308', background: '#fef08a' }, font: { color: '#000' } });
    }
  });

  for (let i = 0; i < pathWords.length - 1; i++) {
    const id1 = `${pathWords[i]}-${pathWords[i + 1]}`;
    const id2 = `${pathWords[i + 1]}-${pathWords[i]}`;
    let e = edges.get(id1) || edges.get(id2);
    if (e) {
      originalEdges.push({ ...e });
      edges.update({ id: e.id, width: 4, color: { color: '#eab308' } });
    }
  }

  // Restore defaults after 4 seconds
  setTimeout(() => {
    nodes.update(originalNodes);
    edges.update(originalEdges);
  }, 4000);
}

// Initial Setup
function setup() {
  initNetwork();

  // Search logic
  const doSearch = () => {
    const val = searchInput.value.trim().toLowerCase();
    if (val) {
      expandWord(val, true);
    }
  };

  const doReset = () => {
    nodes.clear();
    edges.clear();
    loadedWords.clear();
    wordAttributes.clear();
    infoPanel.classList.add('hidden');
    searchInput.value = '';
    expandWord('intelligence', true);
  };

  searchBtn.addEventListener('click', doSearch);
  resetBtn.addEventListener('click', doReset);
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') doSearch();
  });

  // Delete selected nodes (+ cascade children on out-edges) with Delete/Backspace key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
      // Don't delete if user is typing in search box
      if (document.activeElement === searchInput) return;

      const selected = network.getSelectedNodes();
      if (selected.length > 0) {
        // Collect all nodes to delete (selected + their out-edge subtrees)
        const toDelete = new Set();
        const collectCascade = (nodeId) => {
          if (toDelete.has(nodeId)) return;
          toDelete.add(nodeId);
          // Find outgoing edges (from === nodeId) and cascade
          const connectedEdges = network.getConnectedEdges(nodeId);
          connectedEdges.forEach(edgeId => {
            const edge = edges.get(edgeId);
            if (edge && edge.from === nodeId) {
              collectCascade(edge.to);
            }
          });
        };

        selected.forEach(nodeId => collectCascade(nodeId));

        // Remove all collected nodes and their edges
        toDelete.forEach(nodeId => {
          const connectedEdges = network.getConnectedEdges(nodeId);
          edges.remove(connectedEdges);
          nodes.remove(nodeId);
          loadedWords.delete(nodeId);
        });
        infoPanel.classList.add('hidden');
      }
    }
  });

  // Help panel toggle
  const helpBtn = document.getElementById('helpBtn');
  const helpOverlay = document.getElementById('helpOverlay');
  const helpCloseBtn = document.getElementById('helpCloseBtn');

  helpBtn.addEventListener('click', () => helpOverlay.classList.remove('hidden'));
  helpCloseBtn.addEventListener('click', () => helpOverlay.classList.add('hidden'));
  helpOverlay.addEventListener('click', (e) => {
    if (e.target === helpOverlay) helpOverlay.classList.add('hidden');
  });

  // Seed with an initial word
  expandWord('intelligence', true);
}

// Start app
window.addEventListener('DOMContentLoaded', setup);
