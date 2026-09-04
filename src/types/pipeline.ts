// Shared types for the Master Recap Pipeline app

export type JobStatus =
  | "pending"
  | "scraping"
  | "transcribing"
  | "translating"
  | "rendering"
  | "merging"
  | "done"
  | "error"
  | "cancelled";

export type Stage =
  | "search"
  | "scrape"
  | "transcribe"
  | "translate"
  | "slice"
  | "narrate"
  | "tts"
  | "captions"
  | "render"
  | "merge"
  | "bgm"
  | "done";

export type MangaSource =
  | "mangahere"
  | "fanfox"
  | "webtoons"
  | "mal"
  | "anilist"
  | "asurascans"
  | "mangadex"
  | "mangapill"
  | "toonily"
  | "comick"
  | "weebcentral";

export interface MangadexManga {
  id: string;
  title: string;
  description: string;
  coverUrl: string | null;
  status: string | null;
  year: number | null;
  originalLanguage: string | null;
  availableTranslatedLanguages: string[];
  tags: string[];
  contentRating: string | null;
  lastChapter: string | null;
  /** Where this result came from. Defaults to "mangadex" for backward compat. */
  source?: MangaSource;
  /** Link to the original source page (MAL/AniList URL) for non-MangaDex results. */
  externalUrl?: string | null;
  /** Number of followers (from MangaDex or MAL). Optional — used for sorting. */
  followedCount?: number | null;
  /** ISO timestamp of last content update. Optional — used for sorting. */
  updatedAt?: string | null;
}

export interface MangaSearchResult {
  manga: MangadexManga[];
  total: number;
}

export interface ChapterInfo {
  index: number;
  mangadexId: string;
  chapterNum: string | null;
  title: string | null;
  language: string;
  pageCount: number;
  translated: boolean;
  transcribed: boolean;
  rendered: boolean;
  status: string;
  error: string | null;
}

export interface JobDetail {
  id: string;
  mangaId: string;
  mangaTitle: string;
  coverUrl: string | null;
  language: string;
  sourceLang: string | null;
  status: JobStatus;
  progress: number;
  stage: string | null;
  message: string | null;
  totalChapters: number;
  doneChapters: number;
  totalImages: number;
  doneImages: number;
  outputDir: string | null;
  outputVideo: string | null;
  r2Key: string | null;
  archiveProvider: string | null;
  archiveFileId: string | null;
  autoArchive: boolean;
  error: string | null;
  voice: string;
  chapterLimit: number;
  translate: boolean;
  narrate: boolean;
  bgmPath: string | null;
  useBgm: boolean;
  createdAt: string;
  updatedAt: string;
  chapters: ChapterInfo[];
}

export interface JobLogEntry {
  id: string;
  jobId: string;
  level: "info" | "warn" | "error" | "success";
  stage: string | null;
  message: string;
  createdAt: string;
}

// WebSocket events (client -> server)
export type ClientEvent =
  | { type: "subscribe"; jobId: string }
  | { type: "unsubscribe"; jobId: string }
  | { type: "cancel"; jobId: string };

// WebSocket events (server -> client)
export type ServerEvent =
  | { type: "subscribed"; jobId: string }
  | { type: "status"; job: JobDetail }
  | { type: "log"; log: JobLogEntry }
  | { type: "progress"; jobId: string; progress: number; doneChapters: number; totalChapters: number; doneImages: number; totalImages: number; stage: string; message: string }
  | { type: "chapter"; jobId: string; chapter: ChapterInfo }
  | { type: "done"; jobId: string; outputVideo: string | null }
  | { type: "error"; jobId: string; error: string }
  | { type: "cancelled"; jobId: string };

export interface CreateJobInput {
  mangaId: string;
  mangaTitle: string;
  coverUrl: string | null;
  language: string; // requested language code, e.g. "en", "ko", "ja"
  chapterLimit: number; // 0 = all
  voice: string;
  translate: boolean;
  narrate?: boolean;
  groqKey?: string;
  geminiKey?: string;
  openRouterKey?: string;
  zhipuKey?: string;
  siliconFlowKey?: string;
  openaiKey?: string;
  megaEmail?: string;
  megaPassword?: string;
  r2AccountId?: string;
  r2AccessKeyId?: string;
  r2SecretAccessKey?: string;
  r2Bucket?: string;
  autoArchive?: boolean;
  bgmPath?: string | null; // BGM filename (relative to data/bgm/), null = default
  useBgm?: boolean; // whether to overlay BGM
  chapterIds?: string[]; // specific chapter IDs to process (overrides chapterLimit)
}

export interface BgmTrack {
  name: string;
  size: number;
  isDefault: boolean;
}

export interface AppSettings {
  groqKey: string;
  geminiKey: string;
  openRouterKey: string;
  zhipuKey: string;
  siliconFlowKey: string;
  openaiKey: string;
  megaEmail: string;
  megaPassword: string;
  r2AccountId: string;
  r2AccessKeyId: string;
  r2SecretAccessKey: string;
  r2Bucket: string;
  autoArchive: boolean;
  defaultVoice: string;
  defaultLanguage: string;
  defaultChapterLimit: number;
}
