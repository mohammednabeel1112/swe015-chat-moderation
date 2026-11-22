const get = (id) => document.getElementById(id);
const descriptions = {
  int8: "Quantized BERT runs locally on the CPU.",
  bert: "The same fine-tuned checkpoint runs in full precision.",
  neural: "A neural classifier reads the average of learned word vectors.",
  tfidf: "Logistic regression scores weighted words and word pairs.",
  rules: "The lexicon matches exact words and single-character edits.",
};
let names = {};
let messageCount = 0;
let busy = false;
function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function refresh() {
  get("counter").textContent = `${get("message").value.length} / 500`;
  get("send").disabled =
    busy || !get("message").value.trim() || !names[get("model").value];
}
async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw Error();
    names = await response.json();
    for (const [id, name] of Object.entries(names))
      get("model").add(new Option(name, id));
    get("model").value = "int8";
    get("model").disabled = false;
    get("modelnote").textContent = descriptions.int8;
    refresh();
  } catch {
    get("error").textContent = "Could not reach the models. Refresh to retry.";
  }
}
get("model").addEventListener("change", () => {
  get("modelnote").textContent = descriptions[get("model").value];
  refresh();
});
get("message").addEventListener("input", refresh);
get("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = get("message"),
    text = input.value.trim(),
    model = get("model").value;
  if (busy || !text || !names[model]) return;
  busy = true;
  refresh();
  get("error").textContent = "";
  try {
    const resp = await fetch("/api/moderate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    if (!resp.ok) throw Error();
    const result = await resp.json();
    get("empty")?.remove();
    get("nolog")?.remove();
    messageCount++;
    const item = element("div", "", result.label ? "entry flag" : "entry");
    item.append(element("p", result.display));
    const meta = element("div", "", "meta");
    meta.append(
      element("strong", result.label ? "Offensive" : "Clean"),
      element("span", `${(result.confidence * 100).toFixed(1)}% confidence`),
      element("span", names[result.model]),
    );
    item.append(meta);
    get("feed").append(item);
    get("feed").scrollTop = get("feed").scrollHeight;
    const logEntry = element("li", "");
    logEntry.append(
      element("span", `#${messageCount}  ${names[result.model]}`),
      element("span", `${result.ms.toFixed(2)} ms`, "latency"),
    );
    get("log").prepend(logEntry);
    if (input.value.trim() === text) input.value = "";
    input.focus();
  } catch {
    get("error").textContent = "The message could not be scored. Try again.";
  } finally {
    busy = false;
    refresh();
  }
});
loadModels();
