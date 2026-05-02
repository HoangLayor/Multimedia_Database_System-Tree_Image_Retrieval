const QUERY = process.argv[2]?.trim() || "single tree";
const PER_PAGE = 24;
const START_PAGE = 1;
const END_PAGE = 25;
const slug = QUERY.toLowerCase().replace(/\s+/g, "-").replace(/[^a-z0-9-]/g, "");
const OUTPUT_FILE = `pexels-${slug}.json`;
const SLIM_OUTPUT_FILE = `pexels-${slug}-slim.json`;
const DELAY_MS = 300;

console.log(`Query: "${QUERY}" → ${OUTPUT_FILE}, ${SLIM_OUTPUT_FILE}`);

const HEADERS: Record<string, string> = {
  accept: "*/*",
  "accept-language": "en-US,en;q=0.9,vi;q=0.8",
  "cache-control": "no-cache",
  "content-type": "application/json",
  pragma: "no-cache",
  priority: "u=1, i",
  referer: `https://www.pexels.com/search/${encodeURIComponent(QUERY)}/`,
  "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
  "sec-ch-ua-mobile": "?0",
  "sec-ch-ua-platform": '"macOS"',
  "sec-fetch-dest": "empty",
  "sec-fetch-mode": "cors",
  "sec-fetch-site": "same-origin",
  "secret-key": "H2jk9uKnhRmL6WPwh89zBezWvr",
  "user-agent":
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
  "x-client-type": "react",
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function fetchPage(page: number): Promise<any[]> {
  const url = `https://www.pexels.com/en-us/api/v3/search/photos?query=${encodeURIComponent(QUERY)}&page=${page}&per_page=${PER_PAGE}&seo_tags=true`;
  const res = await fetch(url, { headers: HEADERS });
  if (!res.ok) {
    throw new Error(`page ${page} failed: HTTP ${res.status} ${res.statusText}`);
  }
  const json = (await res.json()) as { data: any[] };
  return json.data;
}

async function fetchPageWithRetry(page: number): Promise<any[]> {
  try {
    return await fetchPage(page);
  } catch (err) {
    console.error(`  retrying page ${page} after error:`, (err as Error).message);
    await sleep(1000);
    return await fetchPage(page);
  }
}

async function main() {
  const allItems: any[] = [];
  for (let page = START_PAGE; page <= END_PAGE; page++) {
    try {
      const items = await fetchPageWithRetry(page);
      allItems.push(...items);
      console.log(`[page ${page}/${END_PAGE}] got ${items.length} items (total ${allItems.length})`);
    } catch (err) {
      console.error(`[page ${page}/${END_PAGE}] giving up:`, (err as Error).message);
      break;
    }
    if (page < END_PAGE) await sleep(DELAY_MS);
  }

  await Bun.write(OUTPUT_FILE, JSON.stringify(allItems, null, 2));
  console.log(`\nSaved ${allItems.length} items to ${OUTPUT_FILE}`);

  const slim = allItems.map((item) => ({
    tags: item.attributes?.tags,
    title: item.attributes?.title,
    image: item.attributes?.image,
  }));
  await Bun.write(SLIM_OUTPUT_FILE, JSON.stringify(slim, null, 2));
  console.log(`Saved ${slim.length} slim items to ${SLIM_OUTPUT_FILE}`);
}

main();
