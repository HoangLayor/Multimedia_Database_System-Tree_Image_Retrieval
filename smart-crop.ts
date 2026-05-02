/**
 * Smart Crop — Object-aware 1:1 square cropping for tree images.
 *
 * Usage:
 *   bun run smart-crop.ts                      # process all un-cropped downloaded images
 *   bun run smart-crop.ts --ids 1 2 3          # process specific image IDs
 *   bun run smart-crop.ts --limit 10           # process first N un-cropped
 *   bun run smart-crop.ts --reprocess          # re-crop already-cropped images too
 */

import { Database } from "bun:sqlite";
import sharp from "sharp";
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

// ─── Config ───────────────────────────────────────────────────────────
const DB_PATH = process.env.DB_PATH ?? "./images.db";
const IMAGES_DIR = "./images";
const CROPS_DIR = "./images-cropped";
const DETECT_SCRIPT = "./detect_tree.py";
const DETECT_BATCH_SIZE = 20; // images per Python invocation
const BREATHING_ROOM = 1.2; // 20% padding around tree box

// ─── Types ────────────────────────────────────────────────────────────
interface TreeBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface DetectionResult {
  file: string;
  width: number;
  height: number;
  tree_found: boolean;
  tree_box: TreeBox | null;
  tree_coverage: number;
  all_tree_boxes: TreeBox[];
  error?: string;
}

interface CropRect {
  x: number;
  y: number;
  size: number;
}

interface ImageRow {
  id: number;
  local_filename: string;
  crop_filename: string | null;
}

// ─── Database Setup ───────────────────────────────────────────────────
const db = new Database(DB_PATH);
db.exec(`PRAGMA journal_mode = WAL`);

// Ensure crop columns exist
const cols = db.prepare(`PRAGMA table_info(images)`).all() as { name: string }[];
const colNames = new Set(cols.map((c) => c.name));

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

// ─── Smart Crop Algorithm ─────────────────────────────────────────────
function computeSquareCrop(
  imageWidth: number,
  imageHeight: number,
  treeBox: TreeBox | null,
): CropRect {
  if (!treeBox) {
    // Fallback: center crop
    const size = Math.min(imageWidth, imageHeight);
    return {
      x: Math.floor((imageWidth - size) / 2),
      y: Math.floor((imageHeight - size) / 2),
      size,
    };
  }

  const { x: tx, y: ty, w: tw, h: th } = treeBox;

  // Target square size: fit the tree with breathing room
  let targetSize = Math.ceil(Math.max(tw, th) * BREATHING_ROOM);

  // Cap at shorter image dimension
  const maxSize = Math.min(imageWidth, imageHeight);
  if (targetSize > maxSize) {
    targetSize = maxSize;
  }

  // Center the crop on the tree center
  const treeCenterX = tx + tw / 2;
  const treeCenterY = ty + th / 2;

  let cropX = Math.round(treeCenterX - targetSize / 2);
  let cropY = Math.round(treeCenterY - targetSize / 2);

  // Clamp to image boundaries
  cropX = Math.max(0, Math.min(cropX, imageWidth - targetSize));
  cropY = Math.max(0, Math.min(cropY, imageHeight - targetSize));

  return { x: cropX, y: cropY, size: targetSize };
}

// ─── Detection via Python ─────────────────────────────────────────────
async function detectTrees(imagePaths: string[]): Promise<DetectionResult[]> {
  const proc = Bun.spawn(["python3", DETECT_SCRIPT, ...imagePaths], {
    stdout: "pipe",
    stderr: "pipe",
  });

  const [stdout, stderr] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    console.error("Detection stderr:", stderr);
    throw new Error(`detect_tree.py exited with code ${exitCode}`);
  }

  // Log progress from stderr
  const lines = stderr.split("\n").filter((l) => l.includes("/"));
  for (const line of lines) {
    console.log(`  ${line.trim()}`);
  }

  return JSON.parse(stdout) as DetectionResult[];
}

// ─── Crop & Save ──────────────────────────────────────────────────────
async function cropAndSave(
  sourceFile: string,
  crop: CropRect,
  outputFile: string,
): Promise<void> {
  await sharp(sourceFile)
    .extract({
      left: crop.x,
      top: crop.y,
      width: crop.size,
      height: crop.size,
    })
    .jpeg({ quality: 92 })
    .toFile(outputFile);
}

// ─── Main Pipeline ────────────────────────────────────────────────────
async function main() {
  await mkdir(CROPS_DIR, { recursive: true });

  const args = process.argv.slice(2);
  const reprocess = args.includes("--reprocess");

  let imageRows: ImageRow[];

  const idsIdx = args.indexOf("--ids");
  const limitIdx = args.indexOf("--limit");

  if (idsIdx !== -1) {
    const ids = args
      .slice(idsIdx + 1)
      .filter((a) => !a.startsWith("--"))
      .map(Number);
    const placeholders = ids.map(() => "?").join(",");
    imageRows = db
      .prepare(
        `SELECT id, local_filename, crop_filename FROM images
         WHERE id IN (${placeholders})
         AND local_filename IS NOT NULL
         AND deleted_at IS NULL`,
      )
      .all(...ids) as ImageRow[];
  } else {
    const cropCondition = reprocess ? "" : "AND crop_filename IS NULL";
    const limitClause =
      limitIdx !== -1 ? `LIMIT ${Number(args[limitIdx + 1])}` : "";
    imageRows = db
      .prepare(
        `SELECT id, local_filename, crop_filename FROM images
         WHERE local_filename IS NOT NULL
         AND deleted_at IS NULL
         ${cropCondition}
         ORDER BY id
         ${limitClause}`,
      )
      .all() as ImageRow[];
  }

  if (imageRows.length === 0) {
    console.log("No images to process.");
    return;
  }

  console.log(`\n🌳 Smart Crop: processing ${imageRows.length} image(s)...\n`);

  const updateStmt = db.prepare(`
    UPDATE images SET
      crop_filename = ?,
      crop_x = ?,
      crop_y = ?,
      crop_size = ?,
      tree_box_json = ?
    WHERE id = ?
  `);

  let processed = 0;
  let succeeded = 0;
  let failed = 0;
  let noTree = 0;

  // Process in batches
  for (let i = 0; i < imageRows.length; i += DETECT_BATCH_SIZE) {
    const batch = imageRows.slice(i, i + DETECT_BATCH_SIZE);
    const paths = batch.map((r) => join(IMAGES_DIR, r.local_filename));

    // Filter out missing files
    const validBatch: { row: ImageRow; path: string }[] = [];
    for (let j = 0; j < batch.length; j++) {
      if (existsSync(paths[j])) {
        validBatch.push({ row: batch[j], path: paths[j] });
      } else {
        console.log(`  ⚠ ${batch[j].local_filename} not found, skipping`);
        failed++;
      }
    }

    if (validBatch.length === 0) continue;

    console.log(
      `\n📦 Batch ${Math.floor(i / DETECT_BATCH_SIZE) + 1}: detecting trees in ${validBatch.length} images...`,
    );

    let detections: DetectionResult[];
    try {
      detections = await detectTrees(validBatch.map((v) => v.path));
    } catch (e) {
      console.error(`  ✗ Batch detection failed:`, (e as Error).message);
      failed += validBatch.length;
      continue;
    }

    // Map detections by file path
    const detectionMap = new Map<string, DetectionResult>();
    for (const d of detections) {
      detectionMap.set(d.file, d);
    }

    // Crop each image
    for (const { row, path } of validBatch) {
      const detection = detectionMap.get(path);
      if (!detection) {
        console.log(`  ⚠ No detection result for ${row.local_filename}`);
        failed++;
        continue;
      }

      if (detection.error) {
        console.log(
          `  ✗ ${row.local_filename}: detection error: ${detection.error}`,
        );
        failed++;
        continue;
      }

      const crop = computeSquareCrop(
        detection.width,
        detection.height,
        detection.tree_box,
      );

      const cropFilename = `${row.id}_crop.jpg`;
      const cropPath = join(CROPS_DIR, cropFilename);

      try {
        await cropAndSave(path, crop, cropPath);

        updateStmt.run(
          cropFilename,
          crop.x,
          crop.y,
          crop.size,
          detection.tree_box ? JSON.stringify(detection.tree_box) : null,
          row.id,
        );

        if (detection.tree_found) {
          succeeded++;
        } else {
          noTree++;
        }
        processed++;

        if (processed % 50 === 0) {
          console.log(`  ✓ Progress: ${processed}/${imageRows.length}`);
        }
      } catch (e) {
        console.error(
          `  ✗ ${row.local_filename}: crop failed: ${(e as Error).message}`,
        );
        failed++;
      }
    }
  }

  console.log(`\n${"─".repeat(50)}`);
  console.log(`✅ Smart Crop complete!`);
  console.log(`   Total processed: ${processed}`);
  console.log(`   Tree detected:   ${succeeded}`);
  console.log(`   Center-cropped:  ${noTree} (no tree detected)`);
  console.log(`   Failed:          ${failed}`);
  console.log(`   Output dir:      ${CROPS_DIR}/`);
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
