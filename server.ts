import { Database } from "bun:sqlite";
import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, extname } from "node:path";
import { findSlimFiles, seedFiles } from "./seed";

const PORT = Number(process.env.PORT ?? 3000);
const IMAGES_DIR = "./images";
const CROPS_DIR = "./images-cropped";
const DB_PATH = process.env.DB_PATH ?? "./images.db";
const DOWNLOAD_CONCURRENCY = 4;
const DETECT_SCRIPT = "./detect_tree.py";
const BREATHING_ROOM = 1.2;

await mkdir(IMAGES_DIR, { recursive: true });
await mkdir(CROPS_DIR, { recursive: true });

const db = new Database(DB_PATH);
db.exec(`
  PRAGMA journal_mode = WAL;
  PRAGMA foreign_keys = ON;

  CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT,
    download_link   TEXT NOT NULL UNIQUE,
    image_small     TEXT,
    image_medium    TEXT,
    image_large     TEXT,
    local_filename  TEXT,
    downloaded_at   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS image_tags (
    image_id  INTEGER NOT NULL,
    tag       TEXT NOT NULL,
    source    TEXT NOT NULL DEFAULT 'user',
    PRIMARY KEY (image_id, tag),
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
  );
  CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag);
`);

const cols = db.prepare(`PRAGMA table_info(images)`).all() as { name: string }[];
const colNames = new Set(cols.map((c) => c.name));
if (!colNames.has("deleted_at")) {
  db.exec(`ALTER TABLE images ADD COLUMN deleted_at TEXT`);
}
if (!colNames.has("crop_filename")) {
  db.exec(`ALTER TABLE images ADD COLUMN crop_filename TEXT`);
}
if (!colNames.has("crop_x")) {
  db.exec(`ALTER TABLE images ADD COLUMN crop_x INTEGER`);
}
if (!colNames.has("crop_y")) {
  db.exec(`ALTER TABLE images ADD COLUMN crop_y INTEGER`);
}
if (!colNames.has("crop_size")) {
  db.exec(`ALTER TABLE images ADD COLUMN crop_size INTEGER`);
}
if (!colNames.has("tree_box_json")) {
  db.exec(`ALTER TABLE images ADD COLUMN tree_box_json TEXT`);
}
db.exec(`CREATE INDEX IF NOT EXISTS idx_images_deleted_at ON images(deleted_at)`);

async function downloadImage(id: number, downloadLink: string) {
  const res = await fetch(downloadLink, {
    redirect: "follow",
    headers: {
      "user-agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
      referer: "https://www.pexels.com/",
    },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const buf = await res.arrayBuffer();

  const urlObj = new URL(downloadLink);
  const dlName = urlObj.searchParams.get("dl");
  const ext = (dlName && extname(dlName)) || extname(urlObj.pathname) || ".jpg";
  const filename = `${id}${ext.toLowerCase()}`;
  const filepath = join(IMAGES_DIR, filename);
  await Bun.write(filepath, buf);

  db.prepare(
    `UPDATE images SET local_filename = ?, downloaded_at = datetime('now') WHERE id = ?`,
  ).run(filename, id);
  return filename;
}

async function downloadMany(ids: number[]) {
  const results: { id: number; ok: boolean; error?: string; filename?: string }[] = [];
  const queue = [...ids];
  const workers = Array.from({ length: DOWNLOAD_CONCURRENCY }, async () => {
    while (queue.length) {
      const id = queue.shift();
      if (id == null) break;
      const row = db
        .prepare(`SELECT id, download_link, local_filename FROM images WHERE id = ?`)
        .get(id) as { id: number; download_link: string; local_filename: string | null } | undefined;
      if (!row) {
        results.push({ id, ok: false, error: "not found" });
        continue;
      }
      if (row.local_filename) {
        results.push({ id, ok: true, filename: row.local_filename });
        continue;
      }
      try {
        const filename = await downloadImage(row.id, row.download_link);
        results.push({ id, ok: true, filename });
      } catch (e) {
        results.push({ id, ok: false, error: (e as Error).message });
      }
    }
  });
  await Promise.all(workers);
  return results;
}

function imagesWithTags(rows: { id: number }[]) {
  if (rows.length === 0) return [];
  const ids = rows.map((r) => r.id);
  const placeholders = ids.map(() => "?").join(",");
  const tagRows = db
    .prepare(
      `SELECT image_id, tag, source FROM image_tags WHERE image_id IN (${placeholders}) ORDER BY source DESC, tag`,
    )
    .all(...ids) as { image_id: number; tag: string; source: string }[];
  const byId = new Map<number, { tag: string; source: string }[]>();
  for (const t of tagRows) {
    const arr = byId.get(t.image_id) ?? [];
    arr.push({ tag: t.tag, source: t.source });
    byId.set(t.image_id, arr);
  }
  return rows.map((r) => ({ ...r, tags: byId.get(r.id) ?? [] }));
}

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

const server = Bun.serve({
  port: PORT,
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": () => new Response(Bun.file("public/index.html")),
    "/app.js": () => new Response(Bun.file("public/app.js")),
    "/style.css": () => new Response(Bun.file("public/style.css")),

    "/images-cropped/:filename": (req) => {
      const { filename } = req.params;
      if (filename.includes("..") || filename.includes("/")) {
        return new Response("forbidden", { status: 403 });
      }
      const filepath = join(CROPS_DIR, filename);
      if (!existsSync(filepath)) return new Response("not found", { status: 404 });
      return new Response(Bun.file(filepath));
    },

    "/images/:filename": (req) => {
      const { filename } = req.params;
      if (filename.includes("..") || filename.includes("/")) {
        return new Response("forbidden", { status: 403 });
      }
      const filepath = join(IMAGES_DIR, filename);
      if (!existsSync(filepath)) return new Response("not found", { status: 404 });
      return new Response(Bun.file(filepath));
    },

    "/api/stats": () => {
      const row = db
        .prepare(
          `SELECT
             sum(CASE WHEN deleted_at IS NULL THEN 1 ELSE 0 END) AS total,
             sum(CASE WHEN deleted_at IS NULL AND local_filename IS NOT NULL THEN 1 ELSE 0 END) AS downloaded,
             sum(CASE WHEN deleted_at IS NOT NULL THEN 1 ELSE 0 END) AS trashed
           FROM images`,
        )
        .get() as { total: number; downloaded: number; trashed: number };
      return jsonResponse({
        total: row.total ?? 0,
        downloaded: row.downloaded ?? 0,
        trashed: row.trashed ?? 0,
      });
    },

    "/api/seed": {
      POST: async (req) => {
        const url = new URL(req.url);
        const fileParam = url.searchParams.get("file");
        const files = fileParam ? [fileParam] : await findSlimFiles();
        if (files.length === 0) {
          return jsonResponse({ error: "no pexels-*-slim.json files found" }, 400);
        }
        try {
          const result = await seedFiles(db, files);
          return jsonResponse(result);
        } catch (e) {
          return jsonResponse({ error: (e as Error).message }, 500);
        }
      },
    },

    "/api/tags": () => {
      const rows = db
        .prepare(
          `SELECT t.tag, count(*) AS count
             FROM image_tags t
             JOIN images i ON i.id = t.image_id
            WHERE i.deleted_at IS NULL
            GROUP BY t.tag
            ORDER BY count DESC, t.tag`,
        )
        .all() as { tag: string; count: number }[];
      return jsonResponse(rows);
    },

    "/api/images": (req) => {
      const url = new URL(req.url);
      const tag = url.searchParams.get("tag");
      const downloaded = url.searchParams.get("downloaded");
      const search = url.searchParams.get("search");
      const limit = Math.min(Number(url.searchParams.get("limit") ?? 200), 100000);
      const offset = Number(url.searchParams.get("offset") ?? 0);

      const trash = url.searchParams.get("trash") === "1";

      const conds: string[] = [trash ? `deleted_at IS NOT NULL` : `deleted_at IS NULL`];
      const params: any[] = [];

      if (tag) {
        conds.push(`id IN (SELECT image_id FROM image_tags WHERE tag = ?)`);
        params.push(tag);
      }
      if (downloaded === "1") conds.push(`local_filename IS NOT NULL`);
      if (downloaded === "0") conds.push(`local_filename IS NULL`);
      if (search) {
        conds.push(`title LIKE ?`);
        params.push(`%${search}%`);
      }

      const where = conds.length ? `WHERE ${conds.join(" AND ")}` : "";
      const total = (
        db.prepare(`SELECT count(*) AS c FROM images ${where}`).get(...params) as { c: number }
      ).c;
      const rows = db
        .prepare(
          `SELECT id, title, download_link, image_small, image_medium, image_large,
                  local_filename, downloaded_at, crop_filename, crop_x, crop_y, crop_size, tree_box_json
             FROM images ${where}
             ORDER BY id
             LIMIT ? OFFSET ?`,
        )
        .all(...params, limit, offset) as { id: number }[];
      return jsonResponse({ total, items: imagesWithTags(rows) });
    },

    "/api/images/:id": {
      GET: (req) => {
        const id = Number(req.params.id);
        const row = db
          .prepare(
            `SELECT id, title, download_link, image_small, image_medium, image_large,
                    local_filename, downloaded_at, deleted_at FROM images WHERE id = ?`,
          )
          .get(id) as { id: number } | undefined;
        if (!row) return jsonResponse({ error: "not found" }, 404);
        return jsonResponse(imagesWithTags([row])[0]);
      },
      DELETE: (req) => {
        const id = Number(req.params.id);
        const result = db
          .prepare(
            `UPDATE images SET deleted_at = datetime('now') WHERE id = ? AND deleted_at IS NULL`,
          )
          .run(id);
        if (result.changes === 0) {
          const exists = db.prepare(`SELECT 1 FROM images WHERE id = ?`).get(id);
          if (!exists) return jsonResponse({ error: "not found" }, 404);
        }
        return jsonResponse({ ok: true, deleted: result.changes });
      },
    },

    "/api/images/:id/restore": {
      POST: (req) => {
        const id = Number(req.params.id);
        const result = db
          .prepare(`UPDATE images SET deleted_at = NULL WHERE id = ?`)
          .run(id);
        return jsonResponse({ ok: true, restored: result.changes });
      },
    },

    "/api/images/:id/download": {
      POST: async (req) => {
        const id = Number(req.params.id);
        const row = db
          .prepare(`SELECT id, download_link, local_filename FROM images WHERE id = ?`)
          .get(id) as { id: number; download_link: string; local_filename: string | null } | undefined;
        if (!row) return jsonResponse({ error: "not found" }, 404);
        if (row.local_filename) return jsonResponse({ ok: true, filename: row.local_filename });
        try {
          const filename = await downloadImage(row.id, row.download_link);
          return jsonResponse({ ok: true, filename });
        } catch (e) {
          return jsonResponse({ ok: false, error: (e as Error).message }, 500);
        }
      },
    },

    "/api/export/zip": (req) => {
      const url = new URL(req.url);
      const tag = url.searchParams.get("tag");
      const search = url.searchParams.get("search");

      const conds: string[] = [`deleted_at IS NULL`, `local_filename IS NOT NULL`];
      const params: any[] = [];
      if (tag) {
        conds.push(`id IN (SELECT image_id FROM image_tags WHERE tag = ?)`);
        params.push(tag);
      }
      if (search) {
        conds.push(`title LIKE ?`);
        params.push(`%${search}%`);
      }
      const where = `WHERE ${conds.join(" AND ")}`;
      const rows = db
        .prepare(`SELECT local_filename FROM images ${where} ORDER BY id`)
        .all(...params) as { local_filename: string }[];

      if (rows.length === 0) {
        return jsonResponse({ error: "no downloaded images to export" }, 400);
      }

      const filenames = rows.map((r) => r.local_filename);
      const proc = Bun.spawn(["zip", "-q", "-@", "-"], {
        cwd: IMAGES_DIR,
        stdin: "pipe",
        stdout: "pipe",
        stderr: "inherit",
      });
      proc.stdin.write(filenames.join("\n") + "\n");
      proc.stdin.end();

      const ts = new Date().toISOString().slice(0, 10);
      const slug = tag ? tag.toLowerCase().replace(/[^a-z0-9]+/g, "-") : "all";
      const name = `images-${slug}-${ts}.zip`;

      return new Response(proc.stdout, {
        headers: {
          "content-type": "application/zip",
          "content-disposition": `attachment; filename="${name}"`,
        },
      });
    },

    "/api/download-all": {
      POST: async () => {
        const rows = db
          .prepare(
            `SELECT id FROM images WHERE local_filename IS NULL AND deleted_at IS NULL ORDER BY id`,
          )
          .all() as { id: number }[];
        const results = await downloadMany(rows.map((r) => r.id));
        const ok = results.filter((r) => r.ok).length;
        const failed = results.filter((r) => !r.ok);
        return jsonResponse({ ok, failedCount: failed.length, failed });
      },
    },

    "/api/images/:id/tags": {
      POST: async (req) => {
        const id = Number(req.params.id);
        const body = (await req.json().catch(() => null)) as { tag?: string } | null;
        const tag = body?.tag?.trim();
        if (!tag) return jsonResponse({ error: "tag required" }, 400);
        const exists = db.prepare(`SELECT 1 FROM images WHERE id = ?`).get(id);
        if (!exists) return jsonResponse({ error: "image not found" }, 404);
        db.prepare(
          `INSERT OR IGNORE INTO image_tags (image_id, tag, source) VALUES (?, ?, 'user')`,
        ).run(id, tag);
        return jsonResponse({ ok: true, tag });
      },
    },

    "/api/images/:id/tags/:tag": {
      DELETE: (req) => {
        const id = Number(req.params.id);
        const tag = decodeURIComponent(req.params.tag);
        const result = db
          .prepare(`DELETE FROM image_tags WHERE image_id = ? AND tag = ?`)
          .run(id, tag);
        return jsonResponse({ ok: true, removed: result.changes });
      },
    },

    "/api/smart-crop/:id": {
      POST: async (req) => {
        const id = Number(req.params.id);
        const row = db
          .prepare(`SELECT id, local_filename FROM images WHERE id = ? AND local_filename IS NOT NULL`)
          .get(id) as { id: number; local_filename: string } | undefined;
        if (!row) return jsonResponse({ error: "image not found or not downloaded" }, 404);

        const imagePath = join(IMAGES_DIR, row.local_filename);
        if (!existsSync(imagePath)) return jsonResponse({ error: "file missing" }, 404);

        try {
          // Run detection
          const proc = Bun.spawn(["python3", DETECT_SCRIPT, imagePath], {
            stdout: "pipe",
            stderr: "pipe",
          });
          const stdout = await new Response(proc.stdout).text();
          const exitCode = await proc.exited;
          if (exitCode !== 0) throw new Error("detection failed");

          const detections = JSON.parse(stdout) as any[];
          const detection = detections[0];
          if (!detection) throw new Error("no detection result");

          // Compute crop
          const treeBox = detection.tree_box;
          let targetSize: number, cropX: number, cropY: number;

          if (treeBox) {
            targetSize = Math.ceil(Math.max(treeBox.w, treeBox.h) * BREATHING_ROOM);
            const maxSize = Math.min(detection.width, detection.height);
            if (targetSize > maxSize) targetSize = maxSize;
            const cx = treeBox.x + treeBox.w / 2;
            const cy = treeBox.y + treeBox.h / 2;
            cropX = Math.max(0, Math.min(Math.round(cx - targetSize / 2), detection.width - targetSize));
            cropY = Math.max(0, Math.min(Math.round(cy - targetSize / 2), detection.height - targetSize));
          } else {
            targetSize = Math.min(detection.width, detection.height);
            cropX = Math.floor((detection.width - targetSize) / 2);
            cropY = Math.floor((detection.height - targetSize) / 2);
          }

          // Crop with sharp
          const sharp = (await import("sharp")).default;
          const cropFilename = `${id}_crop.jpg`;
          const cropPath = join(CROPS_DIR, cropFilename);
          await sharp(imagePath)
            .extract({ left: cropX, top: cropY, width: targetSize, height: targetSize })
            .jpeg({ quality: 92 })
            .toFile(cropPath);

          // Update DB
          db.prepare(
            `UPDATE images SET crop_filename=?, crop_x=?, crop_y=?, crop_size=?, tree_box_json=? WHERE id=?`
          ).run(cropFilename, cropX, cropY, targetSize, treeBox ? JSON.stringify(treeBox) : null, id);

          return jsonResponse({
            ok: true,
            id,
            tree_found: detection.tree_found,
            tree_box: treeBox,
            crop: { x: cropX, y: cropY, size: targetSize },
            crop_filename: cropFilename,
          });
        } catch (e) {
          return jsonResponse({ ok: false, error: (e as Error).message }, 500);
        }
      },
    },

    "/api/smart-crop-all": {
      POST: async () => {
        const rows = db
          .prepare(
            `SELECT id FROM images
             WHERE local_filename IS NOT NULL AND deleted_at IS NULL AND crop_filename IS NULL
             ORDER BY id`,
          )
          .all() as { id: number }[];

        if (rows.length === 0) {
          return jsonResponse({ ok: true, message: "all images already cropped", total: 0 });
        }

        // Start batch processing in background
        const proc = Bun.spawn(["bun", "run", "smart-crop.ts"], {
          stdout: "pipe",
          stderr: "pipe",
        });

        // Don't await — return immediately
        proc.exited.then((code) => {
          console.log(`Smart crop batch finished with exit code ${code}`);
        });

        return jsonResponse({
          ok: true,
          message: `Started smart crop for ${rows.length} images`,
          total: rows.length,
        });
      },
    },
  },
  fetch() {
    return new Response("not found", { status: 404 });
  },
});

console.log(`Listening on http://localhost:${server.port}`);
