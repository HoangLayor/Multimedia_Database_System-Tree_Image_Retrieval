const state = {
  images: [],
  total: 0,
  tags: [],
  activeTag: null,
  search: "",
  current: null,
};

const els = {
  grid: document.getElementById("grid"),
  tagList: document.getElementById("tagList"),
  stats: document.getElementById("stats"),
  search: document.getElementById("search"),
  seedBtn: document.getElementById("seedBtn"),
  downloadAllBtn: document.getElementById("downloadAllBtn"),
  exportZipBtn: document.getElementById("exportZipBtn"),
  modal: document.getElementById("modal"),
  modalImg: document.getElementById("modalImg"),
  modalTitle: document.getElementById("modalTitle"),
  modalTags: document.getElementById("modalTags"),
  modalDownloadBtn: document.getElementById("modalDownloadBtn"),
  modalDeleteBtn: document.getElementById("modalDeleteBtn"),
  modalSourceLink: document.getElementById("modalSourceLink"),
  addTagForm: document.getElementById("addTagForm"),
  addTagInput: document.getElementById("addTagInput"),
};

async function api(path, init) {
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j.error) msg = j.error;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function thumbUrl(img) {
  if (img.local_filename) return `/images/${img.local_filename}`;
  return img.image_medium ?? img.image_small;
}
function largeUrl(img) {
  if (img.local_filename) return `/images/${img.local_filename}`;
  return img.image_large ?? img.image_medium;
}

function renderTags() {
  const items = [
    `<li class="${!state.activeTag ? "active" : "clear"}" data-tag="">All <span class="count">${state.total}</span></li>`,
    ...state.tags.map(
      (t) =>
        `<li class="${state.activeTag === t.tag ? "active" : ""}" data-tag="${escapeAttr(t.tag)}">${escapeHtml(t.tag)}<span class="count">${t.count}</span></li>`,
    ),
  ];
  els.tagList.innerHTML = items.join("");
}

function renderGrid() {
  if (state.images.length === 0) {
    els.grid.innerHTML = `<div class="empty">Chưa có ảnh nào. Bấm <b>Seed DB</b> để import từ <code>pexels-single-tree-slim.json</code>.</div>`;
    return;
  }
  els.grid.innerHTML = state.images
    .map((img) => {
      const tagsHtml = img.tags
        .slice(0, 6)
        .map(
          (t) =>
            `<span class="tag-chip ${t.source === "user" ? "user" : ""}">${escapeHtml(t.tag)}</span>`,
        )
        .join("");
      const badge = img.local_filename
        ? `<span class="badge local">local</span>`
        : `<span class="badge remote">remote</span>`;
      return `
        <article class="card" data-id="${img.id}">
          <img class="thumb" src="${escapeAttr(thumbUrl(img))}" loading="lazy" alt="" />
          <div class="body">
            <div class="title">${escapeHtml(img.title ?? "")}</div>
            <div class="tags">${tagsHtml}</div>
            <div class="meta">${badge}<span>#${img.id}</span></div>
          </div>
        </article>`;
    })
    .join("");
}

function renderStats(extra) {
  const downloaded = state.images.filter((i) => i.local_filename).length;
  els.stats.textContent = `${state.total} images${extra ? " · " + extra : ""}`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function escapeAttr(s) { return escapeHtml(s); }

async function loadImages() {
  const params = new URLSearchParams();
  if (state.activeTag) params.set("tag", state.activeTag);
  if (state.search) params.set("search", state.search);
  params.set("limit", "100000");
  const data = await api(`/api/images?${params}`);
  state.images = data.items;
  state.total = data.total;
  renderGrid();
  renderStats();
}

async function loadTags() {
  state.tags = await api("/api/tags");
  renderTags();
}

async function loadStats() {
  const s = await api("/api/stats");
  els.stats.textContent = `${s.total} images · ${s.downloaded} downloaded`;
  els.seedBtn.disabled = s.total > 0;
}

function openModal(img) {
  state.current = img;
  els.modalImg.src = largeUrl(img);
  els.modalTitle.textContent = img.title ?? "";
  els.modalSourceLink.href = img.download_link;
  els.modalDownloadBtn.textContent = img.local_filename ? "Re-fetch" : "Download";
  els.modalDownloadBtn.disabled = !!img.local_filename;
  renderModalTags();
  els.modal.classList.remove("hidden");
}
function closeModal() {
  els.modal.classList.add("hidden");
  state.current = null;
}
function renderModalTags() {
  const img = state.current;
  if (!img) return;
  els.modalTags.innerHTML = img.tags
    .map(
      (t) =>
        `<span class="tag-chip ${t.source === "user" ? "user" : ""}">${escapeHtml(t.tag)}<button data-remove="${escapeAttr(t.tag)}" title="Remove">×</button></span>`,
    )
    .join("");
}

els.tagList.addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  const tag = li.dataset.tag;
  state.activeTag = tag || null;
  renderTags();
  loadImages();
});

els.grid.addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;
  const id = Number(card.dataset.id);
  const img = state.images.find((i) => i.id === id);
  if (img) openModal(img);
});

els.modal.addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeModal();
});

let searchTimer = null;
els.search.addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.search = e.target.value.trim();
    loadImages();
  }, 200);
});

els.seedBtn.addEventListener("click", async () => {
  els.seedBtn.disabled = true;
  els.seedBtn.textContent = "Seeding…";
  try {
    const r = await api("/api/seed", { method: "POST" });
    alert(`Inserted ${r.inserted} images (${r.skipped} skipped)`);
    await Promise.all([loadImages(), loadTags(), loadStats()]);
  } catch (e) {
    alert("Seed failed: " + e.message);
  } finally {
    els.seedBtn.textContent = "Seed DB";
  }
});

els.downloadAllBtn.addEventListener("click", async () => {
  if (!confirm("Download tất cả ảnh chưa có local? Có thể mất vài phút.")) return;
  els.downloadAllBtn.disabled = true;
  els.downloadAllBtn.textContent = "Downloading…";
  try {
    const r = await api("/api/download-all", { method: "POST" });
    alert(`Downloaded ${r.ok}, failed ${r.failedCount}`);
    await Promise.all([loadImages(), loadStats()]);
  } catch (e) {
    alert("Download failed: " + e.message);
  } finally {
    els.downloadAllBtn.textContent = "Download all";
    els.downloadAllBtn.disabled = false;
  }
});

els.exportZipBtn.addEventListener("click", () => {
  const localCount = state.images.filter((i) => i.local_filename).length;
  if (localCount === 0) {
    alert("Chưa có ảnh nào được tải về local. Bấm 'Download all' trước.");
    return;
  }
  const params = new URLSearchParams();
  if (state.activeTag) params.set("tag", state.activeTag);
  if (state.search) params.set("search", state.search);
  const qs = params.toString();
  const scope = state.activeTag ? `tag "${state.activeTag}"` : "tất cả";
  if (!confirm(`Tải ZIP ${localCount} ảnh (${scope}) về máy?`)) return;
  window.location.href = `/api/export/zip${qs ? "?" + qs : ""}`;
});

els.modalDeleteBtn.addEventListener("click", async () => {
  if (!state.current) return;
  if (!confirm(`Xoá ảnh #${state.current.id}? (xoá mềm — có thể khôi phục qua API /restore)`)) return;
  els.modalDeleteBtn.disabled = true;
  try {
    await api(`/api/images/${state.current.id}`, { method: "DELETE" });
    closeModal();
    await Promise.all([loadImages(), loadTags(), loadStats()]);
  } catch (e) {
    alert("Delete failed: " + e.message);
  } finally {
    els.modalDeleteBtn.disabled = false;
  }
});

els.modalDownloadBtn.addEventListener("click", async () => {
  if (!state.current) return;
  els.modalDownloadBtn.disabled = true;
  els.modalDownloadBtn.textContent = "Downloading…";
  try {
    await api(`/api/images/${state.current.id}/download`, { method: "POST" });
    const fresh = await api(`/api/images/${state.current.id}`);
    state.current = fresh;
    const idx = state.images.findIndex((i) => i.id === fresh.id);
    if (idx >= 0) state.images[idx] = fresh;
    renderGrid();
    openModal(fresh);
    loadStats();
  } catch (e) {
    alert("Download failed: " + e.message);
    els.modalDownloadBtn.disabled = false;
  } finally {
    els.modalDownloadBtn.textContent = "Download";
  }
});

els.addTagForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.current) return;
  const tag = els.addTagInput.value.trim();
  if (!tag) return;
  await api(`/api/images/${state.current.id}/tags`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tag }),
  });
  els.addTagInput.value = "";
  const fresh = await api(`/api/images/${state.current.id}`);
  state.current = fresh;
  const idx = state.images.findIndex((i) => i.id === fresh.id);
  if (idx >= 0) state.images[idx] = fresh;
  renderModalTags();
  renderGrid();
  loadTags();
});

els.modalTags.addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-remove]");
  if (!btn || !state.current) return;
  const tag = btn.dataset.remove;
  await api(`/api/images/${state.current.id}/tags/${encodeURIComponent(tag)}`, {
    method: "DELETE",
  });
  const fresh = await api(`/api/images/${state.current.id}`);
  state.current = fresh;
  const idx = state.images.findIndex((i) => i.id === fresh.id);
  if (idx >= 0) state.images[idx] = fresh;
  renderModalTags();
  renderGrid();
  loadTags();
});

loadStats();
loadTags();
loadImages();
