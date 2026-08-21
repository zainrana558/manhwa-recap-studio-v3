/**
 * mega-storage.ts — Mega storage helper (free 20 GB tier).
 *
 * Auth: email + password. Create a free account at mega.nz, then set:
 *   MEGA_EMAIL, MEGA_PASSWORD
 *
 * Files are uploaded to the user's Mega account. A share URL (containing the
 * decryption key) is stored in the DB so downloads can be done anonymously
 * without re-authenticating each time.
 *
 * Uses the `megajs` library (unofficial but widely used and stable).
 */

import { Storage, File } from "megajs";
import type { Storage as MegaStorage } from "megajs";
import { createReadStream, createWriteStream } from "fs";
import type { EventEmitter } from "events";

let storage: MegaStorage | null = null;

export function isMegaConfigured(): boolean {
  return Boolean(process.env.MEGA_EMAIL && process.env.MEGA_PASSWORD);
}

function getMegaStorage(): Promise<MegaStorage> {
  return new Promise((resolve, reject) => {
    if (storage) {
      resolve(storage);
      return;
    }
    const s = new Storage({
      email: process.env.MEGA_EMAIL!,
      password: process.env.MEGA_PASSWORD!,
      autoload: true,
    });

    s.on("ready", () => {
      storage = s;
      resolve(s);
    });
    // megajs's typed `Storage.on()` overloads don't include "error"; fall back
    // to the underlying EventEmitter signature to register the handler.
    (s as unknown as EventEmitter).on("error", (err: Error) => {
      reject(new Error(`Mega login failed: ${err.message}`));
    });
  });
}

/**
 * Upload a file to Mega and return a share URL.
 * The share URL contains both the file handle and the decryption key, so it
 * can be used to download anonymously without logging in again.
 *
 * @param filePath - absolute local path to the file
 * @param filename - the name to give the file in Mega
 * @returns the Mega share URL (e.g. "https://mega.nz/file/xxxx#yyyy")
 */
export async function uploadToMega(
  filePath: string,
  filename: string
): Promise<string> {
  const s = await getMegaStorage();

  return new Promise((resolve, reject) => {
    const uploadStream = s.upload(filename);
    const source = createReadStream(filePath);
    source.pipe(uploadStream as unknown as NodeJS.WritableStream);

    uploadStream.on("complete", () => {
      try {
        const url = (uploadStream as unknown as { link: () => string }).link();
        resolve(url);
      } catch (err) {
        reject(new Error(`Failed to get Mega share URL: ${err}`));
      }
    });
    uploadStream.on("error", (err: Error) => {
      reject(new Error(`Mega upload failed: ${err.message}`));
    });
  });
}

/**
 * Download a file from Mega to a local temp path.
 * Uses the share URL (no login required — anonymous download).
 *
 * @param shareUrl - the Mega share URL stored in the DB
 * @param destPath - where to save the downloaded file
 */
export async function downloadFromMega(
  shareUrl: string,
  destPath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const file = File.fromURL(shareUrl);

    file.loadAttributes((err: Error | null) => {
      if (err) {
        reject(new Error(`Mega loadAttributes failed: ${err.message}`));
        return;
      }

      const dest = createWriteStream(destPath);
      const stream = file.download({});

      stream.pipe(dest);
      stream.on("end", () => resolve());
      stream.on("error", (err: Error) => {
        reject(new Error(`Mega download failed: ${err.message}`));
      });
      dest.on("error", (err: Error) => {
        reject(new Error(`Mega write failed: ${err.message}`));
      });
    });
  });
}
