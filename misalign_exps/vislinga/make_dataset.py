import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, unquote

import pillow_avif  # noqa: F401, needed to register AVIF opener
import cairosvg
import pillow_heif
import requests
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from vislinga.consts import USER_AGENT, AUTH, DATASET_PATH
from PIL import Image, ImageOps, UnidentifiedImageError, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

MAX_PIXELS = 1_000_000_000

try:
    pillow_heif.register_heif_opener()
except Exception:
    pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# Base API endpoint for Commons
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": USER_AGENT
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".tif", ".tiff", ".bmp", ".heic", ".heif", ".avif"
}

retry_with_jitter = retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=0.5, max=10),
    reraise=True,
)

# ----- Cache -----
CACHE_PATH = Path(__file__).parent / "wikimedia_cache.json"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)
            return {}
    return {}


def save_cache(cache: dict) -> None:
    tmp = CACHE_PATH.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(CACHE_PATH)


# ----- Utility: determine if license is CC0 or CC-BY -----
def _license_allows_cc0_or_ccby(extmeta: dict) -> bool:
    if not extmeta:
        return False

    license_url = extmeta.get("LicenseUrl")
    if license_url:
        lu = license_url['value'].lower()
        if "creativecommons.org/publicdomain/zero" in lu:
            return True
        if "creativecommons.org/licenses/by/" in lu and "creativecommons.org/licenses/by-sa/" not in lu:
            return True

    license_short = extmeta["LicenseShortName"]['value']
    assert license_short is not None and isinstance(license_short, str)

    ls = license_short.lower()
    valid_vals = {'cc0', 'public domain'}
    if ls in valid_vals:
        return True

    if 'cc by' in ls and 'cc by-sa' not in ls:
        return True

    return False


@retry_with_jitter
def _call_mediawiki_api(params: dict, timeout: float = 15.0) -> dict:
    resp = requests.get(WIKIMEDIA_API, params=params, headers=HEADERS, auth=AUTH, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data


def fetch_image_urls_for_term(word: str, per_page: int = 10, per_page_mul: int = 3) -> tuple[list, list]:
    """
    Search Wikimedia Commons for `word` and return (urls_list, metadata_list).
    Steps:
    1) Use action=query&list=search with srsearch on namespace 6 (File:)
    2) For found pageids, request prop=imageinfo&iiprop=url|extmetadata to get URLs and license metadata
    3) If > 40 pageids, split into multiple requests and recombine results
    4) Filter to CC0 or CC-BY based on extmetadata
    Returns up to `per_page` results.
    """
    urls: list = []
    meta: list = []

    # 1) search for files (namespace 6 is File:)
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": word,
        "srnamespace": 6,  # File: namespace
        "srlimit": per_page * per_page_mul,  # overfetch to allow license filtering
    }

    try:
        search_data = _call_mediawiki_api(search_params)
    except Exception as e:
        logger.warning("Wikimedia search failed for '%s': %s", word, e)
        return [], []

    search_hits = search_data.get("query", {}).get("search", [])
    if not search_hits:
        return [], []

    # Collect pageids to fetch imageinfo
    pageids = [str(h.get("pageid")) for h in search_hits if h.get("pageid")]
    if not pageids:
        return [], []

    # Helper to chunk a list
    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # 2) fetch imageinfo with extmetadata and url
    # imageinfo accepts up to many pageids; split into chunks if there are many
    MAX_PER_QUERY = 40  # split if > 40
    pages_combined: dict = {}

    # If there are more than MAX_PER_QUERY pageids, make multiple queries and merge results
    pageid_chunks = list(_chunks(pageids, MAX_PER_QUERY))

    for chunk in pageid_chunks:
        imageinfo_params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "pageids": "|".join(chunk),
            "iiprop": "url|extmetadata",
            "iilimit": "max"
        }

        try:
            info_data = _call_mediawiki_api(imageinfo_params)
        except Exception as e:
            # log and continue with other chunks
            logger.warning("Wikimedia imageinfo fetch failed for chunk '%s' (term='%s'): %s", ",".join(chunk), word, e)
            continue

        pages = info_data.get("query", {}).get("pages", {})
        # Ensure keys are strings for consistent lookup later
        for k, v in pages.items():
            pages_combined[str(k)] = v

    # If no pages were fetched at all, return empty
    if not pages_combined:
        return [], []

    # Process pageids in original order using combined_pages
    for pid in pageids:
        page = pages_combined.get(pid, {})
        title = page.get("title")
        ii_list = page.get("imageinfo", [])
        if not ii_list:
            continue
        ii = ii_list[0]
        url = ii.get("url")
        extmeta = ii.get("extmetadata", {})

        if not url or not any(url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            continue

        # check license: accept CC0 or CC-BY
        if not _license_allows_cc0_or_ccby(extmeta):
            continue

        # Build metadata
        # extmeta fields are often nested dicts with 'value' keys
        def _val(k):
            v = extmeta.get(k)
            if isinstance(v, dict):
                return v.get("value")
            return v

        license_name = _val("LicenseShortName") or ""
        license_url = _val("LicenseUrl") or ""
        artist = _val("Artist") or ""
        credit = _val("Credit") or ""
        usage_terms = _val("UsageTerms") or ""

        # source page on commons
        # file pages use /wiki/File:filename
        source_page = f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}" if title else None

        m = {
            "image_url": url,
            "pageid": pid,
            "title": title,
            "license_name": license_name,
            "license_url": license_url,
            "artist": artist,
            "credit": credit,
            "usage_terms": usage_terms,
            "source_page": source_page,
            "query": word,
        }

        urls.append(url)
        meta.append(m)

        if len(urls) >= per_page:
            break

    return urls, meta


def get_search_words() -> set:
    analogy_ds = load_dataset("almogtavor/google-analogy-dataset")
    df = analogy_ds["train"].to_pandas()
    all_words = set(df["Word1"]) | set(df["Word2"]) | set(df["Word3"]) | set(df["Word4"])
    return all_words


def get_image_urls(
        search_words: Iterable[str],
        max_workers: int = 4,
        per_page: int = 10,
) -> tuple[list, list]:
    cache = load_cache()
    words = list(search_words)

    missing_words = [w for w in words if w not in cache or len(cache[w]['urls']) < per_page]

    logger.info(f"Found {len(missing_words)} missing words")

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        for w in missing_words:
            futures[exc.submit(_fetch_safe, w, per_page)] = w

        # Show progress only for the fetches (cached terms are instant)
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Fetching Wikimedia terms",
                           unit="term"):
            word, term_urls, term_meta = future.result()

            # Update cache immediately so we can resume later if interrupted
            cache[word] = {"urls": term_urls, "meta": term_meta}
            try:
                save_cache(cache)
            except Exception as e:
                # Don't fail the whole run if saving the intermediate cache fails
                logger.warning("Failed to save intermediate cache after fetching '%s': %s", word, e)

    # Build output lists in same order as input
    urls_all = []
    meta_all = []
    for word in words:
        entry = cache.get(word, {})
        urls_all.append(entry.get("urls", []))
        meta_all.append(entry.get("meta", []))

    # persist cache one last time (in case nothing was fetched or last save failed)
    try:
        save_cache(cache)
    except Exception as e:
        logger.warning("Failed to save cache at end: %s", e)

    return urls_all, meta_all


def _fetch_safe(word: str, per_page: int = 10, per_page_mul: int = 3, max_per_page_mul: int = 15) \
        -> tuple[str, list, list]:
    """
    Helper to run fetch_image_urls_for_term and always return a 3-tuple:
    (word, urls_list, meta_list). Exceptions are caught and logged.
    """
    try:
        term_urls, term_meta = [], []
        while len(term_urls) < per_page and per_page_mul <= max_per_page_mul:
            urls, meta = fetch_image_urls_for_term(word, per_page_mul=per_page_mul)
            per_page_mul *= 2

            if len(urls) > len(term_urls):
                term_urls = urls
                term_meta = meta
    except Exception as e:
        logger.exception("Uncaught error fetching '%s': %s", word, e)
        term_urls, term_meta = [], []

    return word, term_urls, term_meta


def _safe_filename(s: str, fallback: str = "file") -> str:
    """Make a compact filesystem-safe filename part from s."""
    if not s:
        return fallback
    s = s.strip()
    s = _FILENAME_SAFE_RE.sub("_", s)
    return s[:200]


def _basename_from_url(url: str) -> str:
    """Extract a sensible basename from a URL, or empty string if not available."""
    try:
        path = urlparse(url).path
        name = unquote(path).split("/")[-1]
        return name or ""
    except Exception:
        return ""


@retry_with_jitter
def _download_url_to_path(url: str, path: Path, timeout: float = 30.0) -> None:
    """Download a URL (stream) to the given path, atomically via tmp file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, headers=HEADERS, auth=AUTH, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
    tmp.replace(path)


def _image_dest_and_metadata_name(out_dir: Path, term: str, meta_entry: dict, url: str) -> tuple[Path, Path, Path]:
    """
    Returns (final_jpg_path, meta_path, original_ext_path)
    - final_jpg_path: Path where the JPEG should be written (always .jpg)
    - meta_path: JSON metadata path (final_jpg_path + ".json")
    - original_ext_path: Path with the original extension (what we download first)
    """
    term_dir = out_dir / _safe_filename(term, fallback="term")
    term_dir.mkdir(parents=True, exist_ok=True)

    title = meta_entry.get("title") or ""
    pageid = meta_entry.get("pageid") or ""
    base_from_title = _safe_filename(title) if title else ""
    base_from_url = _safe_filename(_basename_from_url(url)) if url else ""
    base = base_from_title or base_from_url or "image"

    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]

    url_basename = _basename_from_url(url)
    orig_ext = ""
    if "." in url_basename:
        orig_ext = "." + url_basename.split(".")[-1].lower()
        if len(orig_ext) > 6 or not re.match(r"^\.[a-z0-9]+$", orig_ext):
            orig_ext = ""
    if not orig_ext:
        orig_ext = ".bin"

    filename_base = f"{pageid}_{base}_{h}" if pageid else f"{base}_{h}"
    filename_base = _safe_filename(filename_base)

    final_filename = f"{filename_base}.jpg"
    final_filename = _safe_filename(final_filename)
    image_path_jpg = term_dir / final_filename

    orig_filename = f"{filename_base}{orig_ext}"
    orig_filename = _safe_filename(orig_filename)
    original_ext_path = term_dir / orig_filename

    meta_path = image_path_jpg.with_suffix(image_path_jpg.suffix + ".json")
    return image_path_jpg, meta_path, original_ext_path


def open_raster_or_vector_image(path: Path) -> Image.Image:
    """
    Open an image from path, handling:
      - SVG via cairosvg
      - HEIC/HEIF via pillow_heif if installed (Pillow can then open)
      - AVIF via pillow_avif/pillow_heif if installed
      - Fallback to PIL.Image.open for most raster formats
    Returns an already-opened PIL Image (not yet normalized).
    Raises Exception with a helpful message on failure.
    """
    suffix = path.suffix.lower()
    # Handle SVG explicitly: render into PNG bytes, then open with PIL
    if suffix == ".svg" or path.name.lower().endswith(".svg"):
        try:
            png_bytes = cairosvg.svg2png(url=str(path), unsafe=True)
            return Image.open(BytesIO(png_bytes))
        except Exception as e:
            raise RuntimeError(f"Failed to render SVG via cairosvg: {e}")

    try:
        img = Image.open(path)
        # Some plugins lazily fail on load; force a load to detect errors early
        img.verify()  # verify only reads header; will require re-open for operations
        # re-open for actual operations because verify() leaves file in unusable state
        img = Image.open(path)
        return img
    except UnidentifiedImageError:
        # HEIF/AVIF openers were already registered at import, so there is nothing else to try
        raise RuntimeError(f"PIL couldn't identify image file '{path}'. "
                           "Install optional plugins (pillow_heif, pillow-avif-plugin, or cairosvg) for more formats.")


def normalize_image(img: Image.Image, pick_first_frame: bool = True) -> Image.Image:
    """
    Normalize image for JPEG saving:
      - Apply EXIF orientation (ImageOps.exif_transpose)
      - If animated/multi-frame, optionally pick first frame
      - Convert palette/CMYK/L/A modes to RGB
      - Composite alpha onto white background (since JPEG doesn't support alpha)
    Returns an RGB PIL.Image suitable for saving as JPEG.
    """
    # Apply EXIF orientation if present
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        # non-fatal
        pass

    # Handle multi-frame images (GIF, WebP, TIFF, etc.)
    try:
        is_animated = getattr(img, "is_animated", False)
        n_frames = getattr(img, "n_frames", 1)
        if is_animated and pick_first_frame and n_frames > 1:
            try:
                img.seek(0)
                frame = img.copy()
                img = frame
            except Exception:
                # fallback: continue with current image object
                pass
    except Exception:
        pass

    # Convert mode to RGB if not already
    mode = img.mode
    if mode in ("RGBA", "LA") or ("transparency" in getattr(img, "info", {})):
        # Alpha present, composite over white
        try:
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1])  # paste using alpha channel
            img = background
        except Exception:
            # fallback: convert to RGB directly
            img = img.convert("RGB")
    elif mode == "P":  # palette
        img = img.convert("RGB")
    elif mode == "CMYK":
        # convert CMYK to RGB
        img = img.convert("RGB")
    elif mode != "RGB":
        # any other mode -> RGB
        img = img.convert("RGB")

    return img


def save_image_as_jpeg(img: Image.Image, dest_jpg: Path, quality: int = 95) -> None:
    """
    Save PIL Image as JPEG to dest_jpg atomically via tmp file.
    """
    tmp = dest_jpg.with_suffix(dest_jpg.suffix + ".tmp")
    try:
        # ensure parent exists
        dest_jpg.parent.mkdir(parents=True, exist_ok=True)
        # Save to tmp bytes first to detect errors
        img.save(tmp, format="JPEG", quality=quality)
        tmp.replace(dest_jpg)
    finally:
        # cleanup any leftover tmp if exists
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def convert_to_jpeg_preserving_flow(
        original_path: Path,
        image_path_jpg: Path,
        max_side: int = 1000,
        quality: int = 95,
) -> None:
    """
    Full conversion pipeline:
      - open image with handlers
      - normalize (orientation, alpha, frames, color spaces)
      - if either dimension > max_side, downscale (preserve aspect ratio)
      - save as JPEG

    Parameters:
      original_path: path to the downloaded file (any ext)
      image_path_jpg: destination .jpg Path
      max_side: if max(width, height) > max_side then resize to fit within max_side
      quality: JPEG quality
    """
    suffix = original_path.suffix.lower()
    if suffix != ".svg":
        try:
            # header-only; doesn't load image pixels
            with Image.open(original_path) as header_img:
                w, h = header_img.size
        except Exception:
            w = h = None

        if w and h:
            if MAX_PIXELS is not None and (w * h) > MAX_PIXELS:
                raise RuntimeError(
                    f"Image size ({w * h} pixels) exceeds set pipeline limit of {MAX_PIXELS} pixels; skipping."
                )

    img = open_raster_or_vector_image(original_path)
    img = normalize_image(img, pick_first_frame=True)

    # Determine whether to downscale
    try:
        w, h = img.size
    except Exception:
        w = h = None

    if w and h:
        max_dimension = max(w, h)
        if max_dimension > max_side:
            # compute new size preserving aspect ratio
            scale = max_side / float(max_dimension)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            try:
                # Use high-quality downsampling filter
                img = img.resize((new_w, new_h), Image.LANCZOS)
            except Exception as e:
                # If resize fails for some reason, continue with original image (non-fatal)
                logger.warning("Failed to downscale image %s -> %sx%s: %s", original_path, new_w, new_h, e)

    save_image_as_jpeg(img, image_path_jpg, quality=quality)


def _download_image_task(term: str, url: str, meta_entry: dict, out_dir: Path) -> tuple[bool, str, str]:
    """
    Download one image and write its metadata JSON.
    Converts the downloaded file into a JPEG and writes metadata next to the JPEG.

    Returns (success, image_path_str or '', error_message or '').
    """
    image_path_jpg, meta_path, original_ext_path = _image_dest_and_metadata_name(out_dir, term, meta_entry, url)

    # Download to original_ext_path (atomic via .tmp inside _download_url_to_path)
    try:
        _download_url_to_path(url, original_ext_path)
    except Exception as e:
        # cleanup possible tmp file produced by _download_url_to_path
        try:
            tmp = original_ext_path.with_suffix(original_ext_path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False, "", f"download failed: {e}"

    # Convert to JPEG (generalized)
    try:
        convert_to_jpeg_preserving_flow(original_ext_path, image_path_jpg)
    except Exception as e:
        err_msg = f"conversion to JPEG failed: {e}"
        logger.warning(err_msg)
        return False, "", err_msg

    # Remove the original ext file if it still exists and is different from the final jpg
    try:
        if original_ext_path.exists() and original_ext_path != image_path_jpg:
            try:
                original_ext_path.unlink()
            except Exception:
                pass
    except Exception:
        pass

    # Write metadata JSON next to the JPG
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            out_meta = {
                "url": url,
                **meta_entry,
            }
            json.dump(out_meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to write metadata for %s: %s", url, e)

    return True, str(image_path_jpg), ""


def download_all_images(
        urls_all: list[list[str]],
        meta_all: list[list[dict]],
        out_dir: str = DATASET_PATH,
        max_workers: int = 16,
) -> list[tuple[str, bool, str, str]]:
    """
    Download all images collected by get_image_urls.

    Returns a list of tuples:
      (term, success_bool, image_path_or_empty, error_or_hint)
    One entry per attempted image.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    tasks = [(meta["query"], url, meta) for term_urls, term_meta in zip(urls_all, meta_all)
             for url, meta in zip(term_urls, term_meta)]

    results = []
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exc:
        for term, url, meta_entry in tasks:
            futures[exc.submit(_download_image_task, term, url, meta_entry, out_dir_path)] = (term, url)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images", unit="img"):
            term, url = futures[future]
            try:
                success, path_or_empty, err = future.result()
            except Exception as e:
                logger.exception("Unexpected exception downloading %s: %s", url, e)
                success, path_or_empty, err = False, "", f"future exception: {e}"
            results.append((term, success, path_or_empty, err))
            if not success:
                logger.warning("Failed to download %s (%s): %s", url, term, err)

    return results


def main():
    all_words = get_search_words()
    logger.info("Found %d unique words in the analogy dataset.", len(all_words))

    urls, metadata = get_image_urls(all_words)
    total = sum(len(u) for u in urls)
    logger.info("Collected %d image URLs across %d terms.", total, len(urls))

    download_results = download_all_images(urls, metadata, out_dir=DATASET_PATH, max_workers=32)

    n_successes = sum(r[1] for r in download_results)
    fail = len(download_results) - n_successes
    logger.info("Downloaded %d images; %d failures.", n_successes, fail)


if __name__ == "__main__":
    main()
