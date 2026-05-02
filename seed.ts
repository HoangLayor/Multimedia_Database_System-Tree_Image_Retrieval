import { Database } from "bun:sqlite";
import { readFile, readdir } from "node:fs/promises";

export type SlimItem = {
  title: string;
  tags: { name: string; search_term: string }[];
  image: {
    small: string;
    medium: string;
    large: string;
    download: string;
    download_link: string;
  };
};

export function seedFromSlim(db: Database, items: SlimItem[]) {
  const insertImage = db.prepare(`
    INSERT OR IGNORE INTO images (title, download_link, image_small, image_medium, image_large)
    VALUES (?, ?, ?, ?, ?)
  `);
  const findId = db.prepare(`SELECT id FROM images WHERE download_link = ?`);
  const insertTag = db.prepare(`
    INSERT OR IGNORE INTO image_tags (image_id, tag, source) VALUES (?, ?, 'pexels')
  `);

  let inserted = 0;
  let skipped = 0;
  const tx = db.transaction((rows: SlimItem[]) => {
    for (const it of rows) {
      const link = it.image?.download_link;
      if (!link) continue;
      const result = insertImage.run(
        it.title ?? null,
        link,
        it.image.small ?? null,
        it.image.medium ?? null,
        it.image.large ?? null,
      );
      if (result.changes > 0) inserted++;
      else skipped++;
      const row = findId.get(link) as { id: number } | undefined;
      if (!row) continue;
      for (const t of it.tags ?? []) {
        if (t?.name) insertTag.run(row.id, t.name);
      }
    }
  });
  tx(items);
  return { inserted, skipped };
}

export async function findSlimFiles(): Promise<string[]> {
  const all = await readdir(".");
  return all.filter((f) => /^pexels-.+-slim\.json$/.test(f)).sort();
}

export async function seedFiles(db: Database, files: string[]) {
  const perFile: Record<string, { inserted: number; skipped: number; items: number }> = {};
  let totalInserted = 0;
  let totalSkipped = 0;
  for (const f of files) {
    const items = JSON.parse(await readFile(f, "utf-8")) as SlimItem[];
    const r = seedFromSlim(db, items);
    perFile[f] = { ...r, items: items.length };
    totalInserted += r.inserted;
    totalSkipped += r.skipped;
  }
  return { totalInserted, totalSkipped, perFile };
}

if (import.meta.main) {
  const dbPath = process.env.DB_PATH ?? "./images.db";
  const db = new Database(dbPath);
  db.exec(`
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

  const args = process.argv.slice(2);
  const files = args.length > 0 ? args : await findSlimFiles();
  if (files.length === 0) {
    console.error("No slim files found (pattern: pexels-*-slim.json)");
    process.exit(1);
  }
  console.log(`Seeding ${files.length} file(s) into ${dbPath}…`);
  const result = await seedFiles(db, files);
  for (const [f, r] of Object.entries(result.perFile)) {
    console.log(`  ${f}: ${r.items} items → +${r.inserted} new, ${r.skipped} dup`);
  }
  console.log(
    `\nTotal: +${result.totalInserted} inserted, ${result.totalSkipped} duplicates skipped`,
  );
  const total = (db.prepare(`SELECT count(*) AS c FROM images`).get() as { c: number }).c;
  console.log(`DB now has ${total} images`);
}
