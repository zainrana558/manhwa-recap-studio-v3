"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import {
  ArrowLeft,
  Loader2,
  Play,
  Globe,
  BookOpen,
  Mic2,
  Key,
  Languages,
  Info,
  Clock,
  Volume2,
  CloudUpload,
  ListChecks,
  List,
  CheckSquare,
  Square,
  Image as ImageIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { VoiceSelector } from "@/components/pipeline/voice-selector";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { MangadexManga, AppSettings } from "@/types/pipeline";

interface MangaConfigProps {
  manga: MangadexManga;
  onBack: () => void;
  onJobCreated: (jobId: string) => void;
}

interface ChapterFeedItem {
  id: string;
  chapter: string | null;
  title: string | null;
  language: string;
  pages: number;
  volume: string | null;
}

// Voice data is now in voice-selector.tsx with grouping & search support

export function MangaConfig({ manga, onBack, onJobCreated }: MangaConfigProps) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [chapters, setChapters] = useState<ChapterFeedItem[]>([]);
  const [chapterLoading, setChapterLoading] = useState(true);
  const [language, setLanguage] = useState("en");
  const [chapterLimit, setChapterLimit] = useState(5);
  const [voice, setVoice] = useState("en-US-ChristopherNeural");
  const [groqKey, setGroqKey] = useState("");
  const [geminiKey, setGeminiKey] = useState("");
  const [openRouterKey, setOpenRouterKey] = useState("");
  const [zhipuKey, setZhipuKey] = useState("");
  const [siliconFlowKey, setSiliconFlowKey] = useState("");
  const [megaEmail, setMegaEmail] = useState("");
  const [megaPassword, setMegaPassword] = useState("");
  const [autoArchive, setAutoArchive] = useState(false);
  const [translate, setTranslate] = useState(false);

  // Chapter selection mode: "first-n" = slider, "specific" = grid picker
  const [chapterSelectionMode, setChapterSelectionMode] = useState<"first-n" | "specific">("first-n");
  const [selectedChapterIds, setSelectedChapterIds] = useState<Set<string>>(new Set());

  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Voice preview state
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const previewVoiceRef = useRef<string | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  // Load saved settings + chapter feed on mount.
  useEffect(() => {
    (async () => {
      try {
        const [settingsRes, chaptersRes] = await Promise.all([
          fetch("/api/settings").then((r) => r.json()),
          fetch(`/api/manga/${manga.id}`).then((r) => r.json()),
        ]);
        const s = settingsRes.settings ?? settingsRes;
        setSettings(s);
        setGroqKey(s.groqKey ?? "");
        setGeminiKey(s.geminiKey ?? "");
        setOpenRouterKey(s.openRouterKey ?? "");
        setZhipuKey(s.zhipuKey ?? "");
        setSiliconFlowKey(s.siliconFlowKey ?? "");
        setMegaEmail(s.megaEmail ?? "");
        setMegaPassword(s.megaPassword ?? "");
        setAutoArchive(s.autoArchive ?? false);
        setVoice(s.defaultVoice ?? "en-US-ChristopherNeural");
        setChapterLimit(s.defaultChapterLimit ?? 5);

        const allChapters: ChapterFeedItem[] = chaptersRes.chapters ?? [];
        setChapters(allChapters);

        // Pick the best language: prefer English, else original, else first available.
        const langs = Array.from(new Set(allChapters.map((c) => c.language)));
        const preferred =
          langs.find((l) => l === "en") ||
          langs.find((l) => l === manga.originalLanguage) ||
          langs[0] ||
          "en";
        setLanguage(preferred);
      } catch {
        // non-fatal
      } finally {
        setChapterLoading(false);
      }
    })();
  }, [manga.id, manga.originalLanguage]);

  const availableLanguages = Array.from(new Set(chapters.map((c) => c.language)));
  const filteredChapters = chapters.filter((c) => c.language === language);
  const totalImages = filteredChapters.reduce((s, c) => s + (c.pages ?? 0), 0);

  // Effective selection depends on mode
  const effectiveSelectedChapters = useMemo(() => {
    if (chapterSelectionMode === "specific") {
      return filteredChapters.filter((c) => selectedChapterIds.has(c.id));
    }
    return chapterLimit === 0
      ? filteredChapters
      : filteredChapters.slice(0, chapterLimit);
  }, [chapterSelectionMode, chapterLimit, filteredChapters, selectedChapterIds]);

  const effectiveLimit = effectiveSelectedChapters.length;
  const selectedTotalImages = effectiveSelectedChapters.reduce((s, c) => s + (c.pages ?? 0), 0);
  // Estimate: ~3 min per chapter (scrape + VLM + TTS + render)
  const estimatedMinutes = Math.max(1, Math.round(effectiveLimit * 3));
  const estimatedVideoDuration = Math.max(2, Math.round(effectiveLimit * 4));

  // Chapter selection helpers (for specific mode)
  const toggleChapter = useCallback((id: string) => {
    setSelectedChapterIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAllChapters = useCallback(() => {
    setSelectedChapterIds(new Set(filteredChapters.map((c) => c.id)));
  }, [filteredChapters]);

  const deselectAllChapters = useCallback(() => {
    setSelectedChapterIds(new Set());
  }, []);

  const quickSelect = useCallback((range: "first-5" | "first-10" | "last-5" | "all") => {
    const ids = filteredChapters.map((c) => c.id);
    let picked: string[];
    switch (range) {
      case "first-5":
        picked = ids.slice(0, 5);
        break;
      case "first-10":
        picked = ids.slice(0, 10);
        break;
      case "last-5":
        picked = ids.slice(-5);
        break;
      case "all":
        picked = ids;
        break;
    }
    setSelectedChapterIds(new Set(picked));
  }, [filteredChapters]);

  // Reset selected chapters when language changes
  useEffect(() => {
    setSelectedChapterIds(new Set());
  }, [language]);

  const handleStart = useCallback(async () => {
    setStarting(true);
    setError(null);
    try {
      // Persist settings.
      await fetch("/api/settings", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          groqKey,
          geminiKey,
          openRouterKey,
          zhipuKey,
          siliconFlowKey,
          megaEmail,
          megaPassword,
          autoArchive,
          defaultVoice: voice,
          defaultLanguage: language,
          defaultChapterLimit: chapterLimit,
        }),
      });

      // Determine chapter payload
      const isSpecificMode = chapterSelectionMode === "specific" && selectedChapterIds.size > 0;
      const chapterPayload = isSpecificMode
        ? { chapterLimit: selectedChapterIds.size, chapterIds: Array.from(selectedChapterIds) }
        : { chapterLimit };

      // Create job.
      const res = await fetch("/api/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          mangaId: manga.id,
          mangaTitle: manga.title,
          coverUrl: manga.coverUrl,
          language,
          ...chapterPayload,
          voice,
          translate,
          groqKey: groqKey || undefined,
          geminiKey: geminiKey || undefined,
          openRouterKey: openRouterKey || undefined,
          zhipuKey: zhipuKey || undefined,
          siliconFlowKey: siliconFlowKey || undefined,
          megaEmail: megaEmail || undefined,
          megaPassword: megaPassword || undefined,
          autoArchive,
          useBgm: false,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Failed to start job (${res.status})`);
      }
      const data = await res.json();
      onJobCreated(data.job.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start");
    } finally {
      setStarting(false);
    }
  }, [groqKey, geminiKey, openRouterKey, zhipuKey, siliconFlowKey, megaEmail, megaPassword, autoArchive, voice, language, chapterLimit, chapterSelectionMode, selectedChapterIds, translate, manga, onJobCreated]);

  // --- Voice preview ---
  // Fetches a short edge-tts sample for the selected voice and plays it.
  // Toggles play/pause if the same voice is already loaded.
  const handlePreview = useCallback(async () => {
    // If we already have this voice loaded, just toggle play/pause.
    if (audioRef.current && previewVoiceRef.current === voice) {
      if (previewPlaying) {
        audioRef.current.pause();
        setPreviewPlaying(false);
      } else {
        try {
          await audioRef.current.play();
          setPreviewPlaying(true);
        } catch {
          setPreviewError("Playback failed — try again.");
        }
      }
      return;
    }

    // Fetch a fresh preview for the current voice.
    setPreviewLoading(true);
    setPreviewError(null);
    setPreviewPlaying(false);

    // Stop + clean up any previously loaded audio.
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }

    try {
      const res = await fetch(`/api/voice-preview?voice=${encodeURIComponent(voice)}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Preview failed (${res.status})`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      previewUrlRef.current = url;

      const audio = new Audio(url);
      audioRef.current = audio;
      previewVoiceRef.current = voice;

      audio.onended = () => setPreviewPlaying(false);
      audio.onerror = () => {
        setPreviewError("Playback failed — the audio could not be decoded.");
        setPreviewPlaying(false);
      };

      await audio.play();
      setPreviewPlaying(true);
    } catch (e) {
      setPreviewError(
        e instanceof Error ? e.message : "Failed to load voice preview."
      );
    } finally {
      setPreviewLoading(false);
    }
  }, [voice, previewPlaying]);

  // When the voice selection changes, stop any playing preview and reset state
  // so the next Preview click fetches the new voice.
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    previewVoiceRef.current = null;
    setPreviewPlaying(false);
    setPreviewError(null);
  }, [voice]);

  // Clean up audio + object URL on unmount.
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  return (
    <section className="max-w-5xl mx-auto space-y-6">
      <Button variant="ghost" size="sm" onClick={onBack} className="text-muted-foreground">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to search
      </Button>

      {/* Manga header */}
      <div className="flex flex-col sm:flex-row gap-6 p-6 rounded-xl border border-border bg-card">
        <div className="w-32 sm:w-40 aspect-[3/4] rounded-lg overflow-hidden bg-muted flex-shrink-0 border border-border">
          {manga.coverUrl ? (
            <img src={manga.coverUrl} alt={manga.title} className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-muted-foreground text-xs">
              No cover
            </div>
          )}
        </div>
        <div className="flex-1 space-y-3 min-w-0">
          <div>
            <h2 className="text-2xl font-bold leading-tight">{manga.title}</h2>
            <div className="flex flex-wrap gap-2 mt-2">
              {manga.year && <Badge variant="secondary">{manga.year}</Badge>}
              {manga.status && <Badge variant="secondary">{manga.status}</Badge>}
              {manga.originalLanguage && (
                <Badge variant="outline">Original: {manga.originalLanguage.toUpperCase()}</Badge>
              )}
              {manga.contentRating && (
                <Badge variant="outline" className="capitalize">{manga.contentRating}</Badge>
              )}
            </div>
          </div>
          {manga.description && (
            <p className="text-sm text-muted-foreground line-clamp-4 leading-relaxed">
              {manga.description}
            </p>
          )}
          {manga.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {manga.tags.slice(0, 8).map((t) => (
                <span key={t} className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                  {t}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Configuration */}
      <div className="p-6 rounded-xl border border-border bg-card space-y-6">
        <div className="flex items-center gap-2">
          <Play className="h-5 w-5 text-primary" />
          <h3 className="text-lg font-semibold">Pipeline Configuration</h3>
        </div>

        {/* Language */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm font-medium">
            <Globe className="h-4 w-4 text-muted-foreground" />
            Source language
          </Label>
          <Select value={language} onValueChange={setLanguage} disabled={chapterLoading}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select language" />
            </SelectTrigger>
            <SelectContent>
              {availableLanguages.length === 0 && !chapterLoading && (
                <SelectItem value="en">English (fallback)</SelectItem>
              )}
              {availableLanguages.map((l) => (
                <SelectItem key={l} value={l}>
                  {new Intl.DisplayNames(["en"], { type: "language" }).of(l) ?? l} ({l.toUpperCase()})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {chapterLoading
              ? "Loading available chapters…"
              : `${filteredChapters.length} chapter(s) available in this language${language !== "en" && translate ? " — will be auto-translated to English" : ""}.`}
          </p>
        </div>

        {/* Chapter limit */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="flex items-center gap-2 text-sm font-medium">
              <BookOpen className="h-4 w-4 text-muted-foreground" />
              Chapters to process
            </Label>
            <Badge variant="secondary" className="font-mono">
              {chapterSelectionMode === "specific"
                ? `${effectiveLimit} / ${filteredChapters.length}`
                : chapterLimit === 0
                  ? "ALL"
                  : `${effectiveLimit} / ${filteredChapters.length}`}
            </Badge>
          </div>

          {/* Mode toggle */}
          <div className="flex gap-1 p-1 rounded-lg bg-muted/50 border border-border/50">
            <button
              type="button"
              onClick={() => setChapterSelectionMode("first-n")}
              className={`flex items-center gap-1.5 flex-1 justify-center px-3 py-1.5 rounded-md text-xs font-medium transition-all ${chapterSelectionMode === "first-n" ? "bg-card shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
            >
              <List className="h-3.5 w-3.5" />
              First N chapters
            </button>
            <button
              type="button"
              onClick={() => setChapterSelectionMode("specific")}
              className={`flex items-center gap-1.5 flex-1 justify-center px-3 py-1.5 rounded-md text-xs font-medium transition-all ${chapterSelectionMode === "specific" ? "bg-card shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"}`}
            >
              <ListChecks className="h-3.5 w-3.5" />
              Select specific chapters
            </button>
          </div>

          {/* First N mode: slider */}
          {chapterSelectionMode === "first-n" && (
            <div className="space-y-3 animate-fade-in-up">
              <Slider
                value={[chapterLimit]}
                onValueChange={([v]) => setChapterLimit(v)}
                min={0}
                max={Math.max(50, filteredChapters.length)}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0 = all chapters</span>
                <span>~{totalImages > 0 ? Math.round((effectiveLimit / Math.max(1, filteredChapters.length)) * totalImages) : 0} images to download</span>
              </div>
            </div>
          )}

          {/* Specific mode: chapter grid */}
          {chapterSelectionMode === "specific" && (
            <div className="space-y-3 animate-fade-in-up">
              {/* Select all / deselect all + quick selects */}
              <div className="flex flex-wrap items-center gap-2">
                <Button type="button" variant="outline" size="sm" onClick={selectAllChapters} className="h-7 text-xs">
                  <CheckSquare className="h-3 w-3 mr-1" />
                  Select all
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={deselectAllChapters} className="h-7 text-xs">
                  <Square className="h-3 w-3 mr-1" />
                  Deselect all
                </Button>
                <div className="w-px h-4 bg-border" />
                <Button type="button" variant="outline" size="sm" onClick={() => quickSelect("first-5")} className="h-7 text-xs">
                  First 5
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={() => quickSelect("first-10")} className="h-7 text-xs">
                  First 10
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={() => quickSelect("last-5")} className="h-7 text-xs">
                  Last 5
                </Button>
                <Button type="button" variant="outline" size="sm" onClick={() => quickSelect("all")} className="h-7 text-xs">
                  All
                </Button>
              </div>

              {/* Selected count */}
              <div className="flex items-center gap-2">
                <Badge variant={effectiveLimit > 0 ? "default" : "secondary"} className="text-xs">
                  {effectiveLimit} of {filteredChapters.length} chapters selected
                </Badge>
                {effectiveLimit > 0 && (
                  <span className="text-xs text-muted-foreground">
                    ~{selectedTotalImages} images
                  </span>
                )}
              </div>

              {/* Chapter grid */}
              {filteredChapters.length > 0 && (
                <div className="max-h-64 overflow-y-auto scrollbar-thin rounded-lg border border-border/50 p-2">
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                    {filteredChapters.map((ch, idx) => {
                      const checked = selectedChapterIds.has(ch.id);
                      return (
                        <button
                          key={ch.id}
                          type="button"
                          onClick={() => toggleChapter(ch.id)}
                          className={`flex items-start gap-2 p-3 rounded-lg border bg-card/50 hover:bg-card/80 transition-all text-left animate-item-in ${checked ? "border-primary/30 bg-primary/5" : "border-border"}`}
                          style={{ animationDelay: `${Math.min(idx * 20, 300)}ms` }}
                        >
                          <Checkbox
                            checked={checked}
                            onCheckedChange={() => toggleChapter(ch.id)}
                            className="mt-0.5 flex-shrink-0"
                            aria-label={`Toggle chapter ${ch.chapter ?? idx + 1}`}
                          />
                          <div className="min-w-0 flex-1 space-y-0.5">
                            <p className="text-xs font-medium leading-tight truncate">
                              Ch. {ch.chapter ?? idx + 1}
                            </p>
                            {ch.title && (
                              <p className="text-[10px] text-muted-foreground leading-tight truncate">
                                {ch.title}
                              </p>
                            )}
                            {ch.pages > 0 && (
                              <p className="text-[10px] text-muted-foreground/70 flex items-center gap-1">
                                <ImageIcon className="h-2.5 w-2.5" />
                                {ch.pages} pages
                              </p>
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Duration estimate card */}
          <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border border-border/50">
            <div className="flex items-center gap-2 flex-1">
              <Clock className="h-4 w-4 text-sky-400 flex-shrink-0" />
              <div className="space-y-0.5">
                <p className="text-xs font-medium">Estimated processing time</p>
                <p className="text-xs text-muted-foreground">
                  ~{estimatedMinutes} min pipeline · ~{estimatedVideoDuration} min video output
                  {chapterSelectionMode === "specific" && effectiveLimit > 0 && (
                    <span className="text-muted-foreground/70"> · ~{selectedTotalImages} images</span>
                  )}
                </p>
              </div>
            </div>
          </div>
          {effectiveLimit > 10 && (
            <p className="text-xs text-amber-400/90 flex items-center gap-1.5">
              <Info className="h-3.5 w-3.5" />
              Processing {effectiveLimit} chapters may take a long time (scraping + VLM + TTS + rendering per chapter).
            </p>
          )}
        </div>

        <Separator />

        {/* Voice */}
        <div className="space-y-2">
          <Label className="flex items-center gap-2 text-sm font-medium">
            <Mic2 className="h-4 w-4 text-muted-foreground" />
            Narration voice
          </Label>
          <VoiceSelector
            value={voice}
            onChange={setVoice}
            previewLoading={previewLoading}
            previewPlaying={previewPlaying}
            onPreview={handlePreview}
            previewError={previewError}
          />
          {previewError ? (
            <p className="text-xs text-destructive flex items-center gap-1.5">
              <Info className="h-3 w-3" />
              {previewError}
            </p>
          ) : previewPlaying ? (
            <p className="text-xs text-muted-foreground flex items-center gap-1.5">
              <Volume2 className="h-3 w-3 animate-pulse text-primary" />
              Preview playing…
            </p>
          ) : (
            <p className="text-xs text-muted-foreground">
              Click the speaker icon to hear a sample of the selected voice.
            </p>
          )}
        </div>

        {/* Translate toggle */}
        <div className="flex items-center justify-between gap-4 p-3 rounded-lg bg-muted/50">
          <div className="space-y-0.5">
            <Label className="flex items-center gap-2 text-sm font-medium">
              <Languages className="h-4 w-4 text-muted-foreground" />
              Auto-translate to English
            </Label>
            <p className="text-xs text-muted-foreground">
              Uses Groq to translate non-English transcriptions before narration.
            </p>
          </div>
          <Switch checked={translate} onCheckedChange={setTranslate} />
        </div>

        {/* VLM API Keys — for panel text transcription */}
        <div className="space-y-3 p-4 rounded-lg bg-muted/30 border border-border">
          <div className="flex items-center gap-2">
            <Key className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">VLM API Keys</span>
            <span className="text-xs text-muted-foreground">(for reading panel text)</span>
          </div>
          <p className="text-xs text-muted-foreground">
            At least one key is recommended for transcription. The pipeline uses them
            in round-robin for speed + reliability. All have free tiers.
          </p>

          {/* Groq */}
          <div className="space-y-1">
            <Label className="text-xs font-medium" htmlFor="groqKey">
              Groq <span className="text-muted-foreground">(fastest — LPU hardware)</span>
            </Label>
            <Input
              id="groqKey"
              type="password"
              value={groqKey}
              onChange={(e) => setGroqKey(e.target.value)}
              placeholder="gsk_…"
              className="font-mono text-sm h-8"
            />
            <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer"
               className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80">
              console.groq.com/keys (free)
            </a>
          </div>

          {/* Gemini */}
          <div className="space-y-1">
            <Label className="text-xs font-medium" htmlFor="geminiKey">
              Google Gemini <span className="text-muted-foreground">(15 req/min free)</span>
            </Label>
            <Input
              id="geminiKey"
              type="password"
              value={geminiKey}
              onChange={(e) => setGeminiKey(e.target.value)}
              placeholder="AIza…"
              className="font-mono text-sm h-8"
            />
            <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener noreferrer"
               className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80">
              aistudio.google.com/apikey (free)
            </a>
          </div>

          {/* OpenRouter */}
          <div className="space-y-1">
            <Label className="text-xs font-medium" htmlFor="openRouterKey">
              OpenRouter <span className="text-muted-foreground">(access free LLaVA, Qwen-VL)</span>
            </Label>
            <Input
              id="openRouterKey"
              type="password"
              value={openRouterKey}
              onChange={(e) => setOpenRouterKey(e.target.value)}
              placeholder="sk-or-…"
              className="font-mono text-sm h-8"
            />
            <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer"
               className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80">
              openrouter.ai/keys (free tier available)
            </a>
          </div>

          {/* Zhipu AI */}
          <div className="space-y-1">
            <Label className="text-xs font-medium" htmlFor="zhipuKey">
              Zhipu AI <span className="text-muted-foreground">(free — GLM-4V-Flash, OCR-optimized)</span>
            </Label>
            <Input
              id="zhipuKey"
              type="password"
              value={zhipuKey}
              onChange={(e) => setZhipuKey(e.target.value)}
              placeholder="..."
              className="font-mono text-sm h-8"
            />
            <a href="https://open.bigmodel.cn" target="_blank" rel="noopener noreferrer"
               className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80">
              open.bigmodel.cn (free signup)
            </a>
          </div>

          {/* SiliconFlow — BEST FREE OPTION */}
          <div className="space-y-1 rounded-md border border-primary/20 bg-primary/5 p-2">
            <Label className="text-xs font-medium flex items-center gap-1.5" htmlFor="siliconFlowKey">
              <span className="inline-flex items-center justify-center rounded bg-primary/20 text-[9px] font-bold text-primary px-1 py-0.5 leading-none">BEST FREE</span>
              SiliconFlow <span className="text-muted-foreground">(free Qwen2.5-VL-7B, 14M tokens/mo)</span>
            </Label>
            <Input
              id="siliconFlowKey"
              type="password"
              value={siliconFlowKey}
              onChange={(e) => setSiliconFlowKey(e.target.value)}
              placeholder="sk-..."
              className="font-mono text-sm h-8"
            />
            <a href="https://cloud.siliconflow.cn" target="_blank" rel="noopener noreferrer"
               className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80">
              cloud.siliconflow.cn (free credits on signup)
            </a>
          </div>
        </div>

        {/* Cloud Storage — for archiving finished videos */}
        <div className="space-y-3 p-4 rounded-lg bg-muted/30 border border-border">
          <div className="flex items-center gap-2">
            <CloudUpload className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">Cloud Storage</span>
            <span className="text-xs text-muted-foreground">(optional — auto-upload finished videos)</span>
          </div>

          {/* Auto-archive toggle */}
          <div className="flex items-center justify-between gap-4 p-2 rounded-lg bg-muted/50">
            <div className="space-y-0.5">
              <span className="text-xs font-medium">Auto-archive to cloud after render</span>
              <p className="text-[10px] text-muted-foreground">Uploads the video + frees local disk space</p>
            </div>
            <Switch checked={autoArchive} onCheckedChange={setAutoArchive} />
          </div>

          {/* Mega credentials */}
          <div className="space-y-2">
            <Label className="text-xs font-medium">
              Mega <span className="text-muted-foreground">(20 GB free)</span>
            </Label>
            <div className="grid grid-cols-2 gap-2">
              <Input
                type="email"
                value={megaEmail}
                onChange={(e) => setMegaEmail(e.target.value)}
                placeholder="mega@email.com"
                className="font-mono text-sm h-8"
              />
              <Input
                type="password"
                value={megaPassword}
                onChange={(e) => setMegaPassword(e.target.value)}
                placeholder="password"
                className="font-mono text-sm h-8"
              />
            </div>
            <a href="https://mega.nz/register" target="_blank" rel="noopener noreferrer"
               className="text-[10px] text-primary underline underline-offset-2 hover:text-primary/80">
              Create a free Mega account →
            </a>
          </div>
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        {/* No chapters available warning */}
        {!chapterLoading && chapters.length === 0 && (
          <div className="p-4 rounded-lg border border-amber-500/30 bg-amber-500/10 space-y-2">
            <div className="flex items-start gap-2">
              <Info className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="text-sm font-medium text-amber-300">No readable chapters available</p>
                <p className="text-xs text-amber-400/80 leading-relaxed">
                  This manga&apos;s chapters are hosted on external sites (MangaDex doesn&apos;t host the images directly), so they can&apos;t be scraped.
                  Try searching for a different version of the same title, or pick a manga that has chapters hosted on MangaDex.
                </p>
              </div>
            </div>
          </div>
        )}
        {!chapterLoading && chapters.length > 0 && filteredChapters.length === 0 && (
          <div className="p-4 rounded-lg border border-amber-500/30 bg-amber-500/10 space-y-2">
            <div className="flex items-start gap-2">
              <Info className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="text-sm font-medium text-amber-300">
                  No chapters in {new Intl.DisplayNames(["en"], { type: "language" }).of(language) ?? language.toUpperCase()}
                </p>
                <p className="text-xs text-amber-400/80">
                  Available languages: {availableLanguages.map((l) => new Intl.DisplayNames(["en"], { type: "language" }).of(l) ?? l.toUpperCase()).join(", ")}. Select a different language above.
                </p>
              </div>
            </div>
          </div>
        )}

        <Button
          size="lg"
          className="w-full font-semibold"
          onClick={handleStart}
          disabled={starting || chapterLoading || filteredChapters.length === 0 || (chapterSelectionMode === "specific" && effectiveLimit === 0)}
        >
          {starting ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Starting pipeline…
            </>
          ) : chapterLoading ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Loading chapters…
            </>
          ) : filteredChapters.length === 0 ? (
            <>
              <Info className="h-5 w-5 mr-2" />
              No chapters to process
            </>
          ) : (
            <>
              <Play className="h-5 w-5 mr-2" />
              Start Recap Pipeline
            </>
          )}
        </Button>
      </div>
    </section>
  );
}

