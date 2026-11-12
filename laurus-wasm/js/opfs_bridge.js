// OPFS (Origin Private File System) bridge for laurus-wasm.
//
// Provides async functions for reading/writing files in the browser's
// private file system. All functions operate on a named subdirectory
// under the OPFS root to isolate different indexes.
//
// File names may contain "/" separators (e.g. "segment0/data.post").
// Each "/" segment is mapped to an OPFS subdirectory so that the flat
// MemoryStorage namespace translates cleanly into the OPFS hierarchy.
//
// These functions are imported by Rust via `#[wasm_bindgen(module = "/js/opfs_bridge.js")]`.

/**
 * Get or create a subdirectory handle under the OPFS root.
 * @param {string} name - Directory name for this index.
 * @returns {Promise<FileSystemDirectoryHandle>}
 */
async function getIndexDir(name) {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(name, { create: true });
}

/**
 * Resolve a possibly nested path (e.g. "seg0/data.post") into
 * a { dir, fileName } pair, creating intermediate directories as needed.
 * @param {FileSystemDirectoryHandle} baseDir
 * @param {string} name - File path, may contain "/" separators.
 * @param {boolean} create - Whether to create intermediate directories.
 * @returns {Promise<{dir: FileSystemDirectoryHandle, fileName: string}>}
 */
async function resolvePath(baseDir, name, create = false) {
  const parts = name.split("/");
  const fileName = parts.pop();
  let dir = baseDir;
  for (const part of parts) {
    dir = await dir.getDirectoryHandle(part, { create });
  }
  return { dir, fileName };
}

/**
 * Initialize an OPFS directory for an index.
 * Returns an opaque handle (FileSystemDirectoryHandle) as a JS object.
 * @param {string} name - Index directory name.
 * @returns {Promise<FileSystemDirectoryHandle>}
 */
export async function opfs_init(name) {
  return getIndexDir(name);
}

/**
 * Recursively list all file names in the index directory.
 * Returns paths relative to the base dir, using "/" as separator.
 * @param {FileSystemDirectoryHandle} dir - Directory handle from opfs_init.
 * @returns {Promise<string[]>}
 */
export async function opfs_list_files(dir) {
  const names = [];

  async function walk(handle, prefix) {
    for await (const [entryName, entryHandle] of handle.entries()) {
      const path = prefix ? `${prefix}/${entryName}` : entryName;
      if (entryHandle.kind === "file") {
        names.push(path);
      } else if (entryHandle.kind === "directory") {
        await walk(entryHandle, path);
      }
    }
  }

  await walk(dir, "");
  return names;
}

/**
 * Check if a file exists in the index directory.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name - File path, may contain "/" separators.
 * @returns {Promise<boolean>}
 */
export async function opfs_file_exists(dir, name) {
  try {
    const { dir: parent, fileName } = await resolvePath(dir, name);
    await parent.getFileHandle(fileName);
    return true;
  } catch {
    return false;
  }
}

/**
 * Read the entire contents of a file as a Uint8Array.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name - File path, may contain "/" separators.
 * @returns {Promise<Uint8Array>}
 */
export async function opfs_read_file(dir, name) {
  const { dir: parent, fileName } = await resolvePath(dir, name);
  const fileHandle = await parent.getFileHandle(fileName);
  const file = await fileHandle.getFile();
  const buffer = await file.arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Write data to a file (creates or overwrites).
 * Intermediate directories are created automatically.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name - File path, may contain "/" separators.
 * @param {Uint8Array} data - File contents.
 * @returns {Promise<void>}
 */
export async function opfs_write_file(dir, name, data) {
  const { dir: parent, fileName } = await resolvePath(dir, name, true);
  const fileHandle = await parent.getFileHandle(fileName, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(data);
  await writable.close();
}

/**
 * Delete a file from the index directory.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name - File path, may contain "/" separators.
 * @returns {Promise<void>}
 */
export async function opfs_delete_file(dir, name) {
  try {
    const { dir: parent, fileName } = await resolvePath(dir, name);
    await parent.removeEntry(fileName);
  } catch {
    // Ignore if file doesn't exist
  }
}
