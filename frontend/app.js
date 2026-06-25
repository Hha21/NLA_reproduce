const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------- state
let state = {
  messages:        [],   // [{role, content}]
  tokens:          [],   // string[]
  tokenIds:        [],   // number[]
  isSpecial:       [],   // bool[]
  assistantStart:  0,    // index of first assistant token
  selectedIndex:   null, // currently-selected token index
};

// ---------------------------------------------------------------- api
async function apiPost(path, body) {
  const res = await fetch(path, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "request failed");
  }
  return res.json();
}

function readSettings() {
  return {
    max_new_tokens: parseInt($("#max-tokens").value, 10),
    temperature:    parseFloat($("#temperature").value),
    top_p:          parseFloat($("#top-p").value),
  };
}

// ---------------------------------------------------------------- chat flow
async function sendMessage() {
  const text = $("#chat-input").value.trim();
  if (!text) return;

  const settings = readSettings();
  $("#send-btn").disabled  = true;
  $("#chat-input").value   = "";

  // optimistic: show user message + assistant placeholder
  state.messages.push({ role: "user", content: text });
  renderMessages({ withPlaceholder: true });

  try {
    const res = await apiPost("/api/chat", {
      messages: state.messages,
      ...settings,
    });
    state.messages       = res.messages;
    state.tokens         = res.tokens;
    state.tokenIds       = res.token_ids;
    state.isSpecial      = res.is_special;
    state.assistantStart = res.assistant_token_start;
    state.selectedIndex  = null;

    renderMessages();
    renderTokens();
    resetExplanation();
  } catch (e) {
    // undo optimistic user message on failure
    state.messages.pop();
    renderMessages();
    explanationError(`Chat failed: ${e.message}`);
  } finally {
    $("#send-btn").disabled = false;
    $("#chat-input").focus();
  }
}

function newChat() {
  state = {
    messages: [], tokens: [], tokenIds: [], isSpecial: [],
    assistantStart: 0, selectedIndex: null,
  };
  $("#chat-input").value = "";
  renderMessages();
  renderTokens();
  resetExplanation();
  $("#chat-input").focus();
}

// ---------------------------------------------------------------- rendering
function renderMessages({ withPlaceholder = false } = {}) {
  const div = $("#messages");
  div.innerHTML = "";

  if (state.messages.length === 0 && !withPlaceholder) {
    const ph = document.createElement("div");
    ph.className   = "placeholder";
    ph.textContent = "No messages yet. Type something and press Enter.";
    div.appendChild(ph);
    return;
  }

  for (const m of state.messages) {
    const el = document.createElement("div");
    el.className   = `message ${m.role}`;
    el.textContent = m.content;
    div.appendChild(el);
  }

  if (withPlaceholder) {
    const el = document.createElement("div");
    el.className   = "message assistant loading";
    el.textContent = "generating";
    div.appendChild(el);
  }

  div.scrollTop = div.scrollHeight;
}

function renderTokens() {
  const div = $("#tokens");
  div.innerHTML = "";
  $("#token-count").textContent =
    `${state.tokens.length} token${state.tokens.length === 1 ? "" : "s"}`;

  if (state.tokens.length === 0) {
    const ph = document.createElement("div");
    ph.className   = "placeholder";
    ph.textContent = "Send a message to populate the model context.";
    div.appendChild(ph);
    return;
  }

  state.tokens.forEach((t, i) => {
    const span = document.createElement("span");
    span.className     = "token";
    if (state.isSpecial[i])              span.classList.add("special");
    if (i >= state.assistantStart)       span.classList.add("in-assistant");
    span.textContent   = t;
    span.dataset.index = i;
    span.title         = `Token ${i}${state.isSpecial[i] ? " (special)" : ""}`;
    span.addEventListener("click", () => analyzeToken(i));
    div.appendChild(span);
  });
}

function updateContextHighlight() {
  const tokens = document.querySelectorAll(".token");
  tokens.forEach((el) =>
    el.classList.remove("selected", "in-context", "loading"),
  );
  if (state.selectedIndex == null) return;
  for (let i = 0; i < state.selectedIndex; i++) {
    tokens[i].classList.add("in-context");
  }
  tokens[state.selectedIndex].classList.add("selected");
}

// ---------------------------------------------------------------- analyse
async function analyzeToken(index) {
  state.selectedIndex = index;
  updateContextHighlight();

  const tokenEl = document.querySelectorAll(".token")[index];
  tokenEl.classList.add("loading");

  $("#selected-token").textContent = `[${index}] "${state.tokens[index]}"`;
  $("#selected-token").classList.remove("muted");
  $("#explanation").textContent = "Generating explanation…";
  $("#explanation").classList.add("muted");
  $("#reconstruction").textContent = "—";
  $("#fve").textContent            = "—";

  try {
    const res = await apiPost("/api/analyze", {
      token_ids: state.tokenIds,
      position:  index,
    });
    $("#explanation").textContent = res.explanation || "(empty)";
    $("#explanation").classList.remove("muted");
    $("#reconstruction").textContent = res.reconstruction.toFixed(3);
    $("#fve").textContent            = res.fve == null ? "n/a" : res.fve.toFixed(3);
  } catch (e) {
    explanationError(`Analyse failed: ${e.message}`);
  } finally {
    tokenEl.classList.remove("loading");
  }
}

function resetExplanation() {
  $("#selected-token").textContent = "No token selected";
  $("#selected-token").classList.add("muted");
  $("#explanation").textContent =
    "Click any token in the middle panel to generate an explanation.";
  $("#explanation").classList.add("muted");
  $("#reconstruction").textContent = "—";
  $("#fve").textContent            = "—";
}

function explanationError(msg) {
  $("#explanation").innerHTML = `<span class="error">${msg}</span>`;
  $("#explanation").classList.remove("muted");
}

// ---------------------------------------------------------------- init
$("#send-btn").addEventListener("click", sendMessage);
$("#new-chat-btn").addEventListener("click", newChat);
$("#settings-toggle").addEventListener("click", () => {
  $("#settings").classList.toggle("hidden");
});

$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

$("#chat-input").focus();
