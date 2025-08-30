// --- Autocomplete with fetch + debounce ---
(function () {
  const input = document.getElementById("searchInput");
  const list = document.getElementById("suggestions");
  const root = document.getElementById("autocomplete-root");

  let controller = null;
  let hoverIndex = -1;

  function clearList() {
    list.innerHTML = "";
    list.hidden = true;
    hoverIndex = -1;
  }

  function render(items) {
    list.innerHTML = "";
    items.forEach((txt, idx) => {
      const li = document.createElement("li");
      li.className = "suggestion-item";
      li.setAttribute("role", "option");
      li.textContent = txt;
      li.addEventListener("mousedown", (e) => {
        e.preventDefault();           // keep focus on input
        input.value = txt;
        clearList();
      });
      li.addEventListener("mouseover", () => setHover(idx));
      list.appendChild(li);
    });
    list.hidden = items.length === 0;
  }

  function setHover(i) {
    const children = Array.from(list.children);
    children.forEach((el, k) => el.setAttribute("aria-selected", String(k === i)));
    hoverIndex = i;
  }

  function debounce(fn, ms) {
    let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  const fetchSuggest = debounce(async (q) => {
    if (controller) controller.abort();
    controller = new AbortController();
    const res = await fetch(`/suggest?q=${encodeURIComponent(q)}`, { signal: controller.signal });
    if (!res.ok) return clearList();
    const data = await res.json();
    render(data);
  }, 200);

  input.addEventListener("input", (e) => {
    const q = e.target.value.trim();
    if (!q) return clearList();
    fetchSuggest(q);
  });

  input.addEventListener("keydown", (e) => {
    const items = Array.from(list.children);
    if (list.hidden || items.length === 0) return;

    if (e.key === "ArrowDown") { e.preventDefault(); setHover(Math.min(hoverIndex + 1, items.length - 1)); }
    if (e.key === "ArrowUp")   { e.preventDefault(); setHover(Math.max(hoverIndex - 1, 0)); }
    if (e.key === "Enter") {
      if (hoverIndex >= 0) {
        e.preventDefault();
        input.value = items[hoverIndex].textContent;
        clearList();
      }
    }
    if (e.key === "Escape") clearList();
  });

  // click outside closes the list
  document.addEventListener("click", (e) => {
    if (!root.contains(e.target)) clearList();
  });
})();

// --- Simple slider that auto-advances every 4s ---
(function () {
  const track = document.getElementById("slides");
  const slider = document.getElementById("slider");
  if (!track || !slider) return;

  let index = 0;
  function step() {
    const card = track.children[0];
    if (!card) return;
    const cardWidth = card.getBoundingClientRect().width + 14; // 14px gap in CSS
    index = (index + 1) % track.children.length;
    track.style.transform = `translateX(${-index * cardWidth}px)`;
  }

  setInterval(step, 4000);
})();
