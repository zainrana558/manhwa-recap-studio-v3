"use client";

import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { Search, Check, Volume2, ChevronDown, X, Pause, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// ── Voice data organized by language group ──
interface VoiceEntry {
  value: string;
  label: string;
  gender: "male" | "female";
}

interface VoiceGroup {
  code: string;
  label: string;
  flag: string;
  voices: VoiceEntry[];
}

const VOICE_GROUPS: VoiceGroup[] = [
  {
    // Local Kokoro-82M — runs on the box, no network / no rate limits / one
    // consistent voice for a whole series. IDs here must match
    // _KOKORO_VOICE_RE in pipeline/master_pipeline.py.
    code: "kokoro",
    label: "Local · Kokoro (neural, offline)",
    flag: "🖥️",
    voices: [
      { value: "am_michael", label: "Michael — warm US male (recommended)", gender: "male" },
      { value: "am_onyx", label: "Onyx — deep US male", gender: "male" },
      { value: "am_fenrir", label: "Fenrir — energetic US male", gender: "male" },
      { value: "am_puck", label: "Puck — bright US male", gender: "male" },
      { value: "am_adam", label: "Adam — US male", gender: "male" },
      { value: "bm_george", label: "George — UK male, gravitas", gender: "male" },
      { value: "bm_lewis", label: "Lewis — UK male", gender: "male" },
      { value: "af_heart", label: "Heart — warm US female", gender: "female" },
      { value: "af_bella", label: "Bella — expressive US female", gender: "female" },
      { value: "af_nicole", label: "Nicole — soft US female", gender: "female" },
      { value: "bf_emma", label: "Emma — UK female", gender: "female" },
    ],
  },
  {
    code: "en-US",
    label: "English (US)",
    flag: "🇺🇸",
    voices: [
      { value: "en-US-AndrewMultilingualNeural", label: "Andrew Multilingual", gender: "male" },
      { value: "en-US-AndrewNeural", label: "Andrew", gender: "male" },
      { value: "en-US-BrianMultilingualNeural", label: "Brian Multilingual", gender: "male" },
      { value: "en-US-BrianNeural", label: "Brian", gender: "male" },
      { value: "en-US-ChristopherNeural", label: "Christopher", gender: "male" },
      { value: "en-US-RogerNeural", label: "Roger", gender: "male" },
      { value: "en-US-SteffanNeural", label: "Steffan", gender: "male" },
      { value: "en-US-AvaMultilingualNeural", label: "Ava Multilingual", gender: "female" },
      { value: "en-US-AvaNeural", label: "Ava", gender: "female" },
      { value: "en-US-EmmaMultilingualNeural", label: "Emma Multilingual", gender: "female" },
      { value: "en-US-EmmaNeural", label: "Emma", gender: "female" },
      { value: "en-US-JennyNeural", label: "Jenny", gender: "female" },
      { value: "en-US-MichelleNeural", label: "Michelle", gender: "female" },
      { value: "en-US-CoraNeural", label: "Cora", gender: "female" },
      { value: "en-US-ElizabethNeural", label: "Elizabeth", gender: "female" },
      { value: "en-US-MonicaNeural", label: "Monica", gender: "female" },
      { value: "en-US-SaraNeural", label: "Sara", gender: "female" },
      { value: "en-US-NancyNeural", label: "Nancy", gender: "female" },
    ],
  },
  {
    code: "en-GB",
    label: "English (UK)",
    flag: "🇬🇧",
    voices: [
      { value: "en-GB-RyanNeural", label: "Ryan", gender: "male" },
      { value: "en-GB-ThomasNeural", label: "Thomas", gender: "male" },
      { value: "en-GB-SoniaNeural", label: "Sonia", gender: "female" },
      { value: "en-GB-LibbyNeural", label: "Libby", gender: "female" },
      { value: "en-GB-MaisieNeural", label: "Maisie", gender: "female" },
    ],
  },
  {
    code: "en-AU",
    label: "English (Australia)",
    flag: "🇦🇺",
    voices: [
      { value: "en-AU-WilliamNeural", label: "William", gender: "male" },
      { value: "en-AU-DarrenNeural", label: "Darren", gender: "male" },
      { value: "en-AU-DuncanNeural", label: "Duncan", gender: "male" },
      { value: "en-AU-AnnetteNeural", label: "Annette", gender: "female" },
      { value: "en-AU-CarlyNeural", label: "Carly", gender: "female" },
      { value: "en-AU-DotNeural", label: "Dot", gender: "female" },
      { value: "en-AU-NatashaNeural", label: "Natasha", gender: "female" },
      { value: "en-AU-EliseNeural", label: "Elise", gender: "female" },
      { value: "en-AU-MadisonNeural", label: "Madison", gender: "female" },
    ],
  },
  {
    code: "en-CA",
    label: "English (Canada)",
    flag: "🇨🇦",
    voices: [
      { value: "en-CA-LiamNeural", label: "Liam", gender: "male" },
      { value: "en-CA-ClaraNeural", label: "Clara", gender: "female" },
    ],
  },
  {
    code: "en-IE",
    label: "English (Ireland)",
    flag: "🇮🇪",
    voices: [
      { value: "en-IE-ConnorNeural", label: "Connor", gender: "male" },
      { value: "en-IE-EmilyNeural", label: "Emily", gender: "female" },
    ],
  },
  {
    code: "en-IN",
    label: "English (India)",
    flag: "🇮🇳",
    voices: [
      { value: "en-IN-NeerjaNeural", label: "Neerja", gender: "female" },
      { value: "en-IN-PrabhatNeural", label: "Prabhat", gender: "male" },
    ],
  },
  {
    code: "en-ZA",
    label: "English (South Africa)",
    flag: "🇿🇦",
    voices: [
      { value: "en-ZA-LukeNeural", label: "Luke", gender: "male" },
      { value: "en-ZA-LeahNeural", label: "Leah", gender: "female" },
    ],
  },
  {
    code: "ja-JP",
    label: "Japanese",
    flag: "🇯🇵",
    voices: [
      { value: "ja-JP-KeitaNeural", label: "Keita", gender: "male" },
      { value: "ja-JP-NanamiNeural", label: "Nanami", gender: "female" },
    ],
  },
  {
    code: "ko-KR",
    label: "Korean",
    flag: "🇰🇷",
    voices: [
      { value: "ko-KR-InJoonNeural", label: "InJoon", gender: "male" },
      { value: "ko-KR-SunHiNeural", label: "Sun-Hi", gender: "female" },
    ],
  },
  {
    code: "zh-CN",
    label: "Chinese",
    flag: "🇨🇳",
    voices: [
      { value: "zh-CN-YunxiNeural", label: "云希", gender: "male" },
      { value: "zh-CN-XiaoxiaoNeural", label: "晓晓", gender: "female" },
    ],
  },
  {
    code: "es-ES",
    label: "Spanish",
    flag: "🇪🇸",
    voices: [
      { value: "es-ES-AlvaroNeural", label: "Álvaro", gender: "male" },
      { value: "es-ES-ElviraNeural", label: "Elvira", gender: "female" },
    ],
  },
  {
    code: "fr-FR",
    label: "French",
    flag: "🇫🇷",
    voices: [
      { value: "fr-FR-HenriNeural", label: "Henri", gender: "male" },
      { value: "fr-FR-DeniseNeural", label: "Denise", gender: "female" },
    ],
  },
  {
    code: "de-DE",
    label: "German",
    flag: "🇩🇪",
    voices: [
      { value: "de-DE-ConradNeural", label: "Conrad", gender: "male" },
      { value: "de-DE-KatjaNeural", label: "Katja", gender: "female" },
    ],
  },
  {
    code: "pt-BR",
    label: "Portuguese (Brazil)",
    flag: "🇧🇷",
    voices: [
      { value: "pt-BR-AntonioNeural", label: "Antonio", gender: "male" },
      { value: "pt-BR-FranciscaNeural", label: "Francisca", gender: "female" },
    ],
  },
  {
    code: "hi-IN",
    label: "Hindi",
    flag: "🇮🇳",
    voices: [
      { value: "hi-IN-MadhurNeural", label: "Madhur", gender: "male" },
      { value: "hi-IN-SwaraNeural", label: "Swara", gender: "female" },
    ],
  },
];

// Build a flat map for quick lookup
const ALL_VOICES = VOICE_GROUPS.flatMap((g) =>
  g.voices.map((v) => ({ ...v, groupCode: g.code, groupLabel: g.label, groupFlag: g.flag }))
);

// Find voice details for a given value
function findVoice(value: string) {
  return ALL_VOICES.find((v) => v.value === value);
}

interface VoiceSelectorProps {
  value: string;
  onChange: (value: string) => void;
  previewLoading?: boolean;
  previewPlaying?: boolean;
  onPreview?: () => void;
  previewError?: string | null;
}

type GenderFilter = "all" | "male" | "female";

export function VoiceSelector({
  value,
  onChange,
  previewLoading,
  previewPlaying,
  onPreview,
  previewError,
}: VoiceSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [genderFilter, setGenderFilter] = useState<GenderFilter>("all");
  const [activeGroup, setActiveGroup] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Reset filters when closing (render-time pattern)
  const [prevOpen, setPrevOpen] = useState(false);
  if (!open && prevOpen) {
    setPrevOpen(false);
    setSearch("");
    setGenderFilter("all");
    setActiveGroup(null);
  }
  if (open && !prevOpen) {
    setPrevOpen(true);
  }

  // Focus search when opening
  useEffect(() => {
    if (open) {
      const id = setTimeout(() => searchInputRef.current?.focus(), 50);
      return () => clearTimeout(id);
    }
  }, [open]);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Escape to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open]);

  // Filtered voice groups
  const filteredGroups = useMemo(() => {
    const q = search.toLowerCase().trim();
    return VOICE_GROUPS.map((group) => {
      let voices = group.voices;

      // Gender filter
      if (genderFilter !== "all") {
        voices = voices.filter((v) => v.gender === genderFilter);
      }

      // Search filter
      if (q) {
        voices = voices.filter(
          (v) =>
            v.label.toLowerCase().includes(q) ||
            v.value.toLowerCase().includes(q) ||
            group.label.toLowerCase().includes(q) ||
            group.code.toLowerCase().includes(q)
        );
      }

      // Active group filter
      if (activeGroup && activeGroup !== group.code) {
        return { ...group, voices: [] };
      }

      return { ...group, voices };
    }).filter((g) => g.voices.length > 0);
  }, [search, genderFilter, activeGroup]);

  const totalFiltered = filteredGroups.reduce((s, g) => s + g.voices.length, 0);
  const selectedVoice = findVoice(value);

  const handleSelect = useCallback(
    (voiceValue: string) => {
      onChange(voiceValue);
      setOpen(false);
    },
    [onChange]
  );

  return (
    <div className="flex gap-2 items-start">
      <div ref={containerRef} className="flex-1 relative">
        {/* Trigger button */}
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className={cn(
            "flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs transition-colors",
            "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
            "hover:bg-accent hover:text-accent-foreground"
          )}
        >
          <span className="truncate">
            {selectedVoice ? (
              <span className="flex items-center gap-1.5">
                <span className="text-xs">{selectedVoice.groupFlag}</span>
                <span>{selectedVoice.label}</span>
                <span className="text-muted-foreground">({selectedVoice.gender === "male" ? "M" : "F"})</span>
              </span>
            ) : (
              <span className="text-muted-foreground">Select voice…</span>
            )}
          </span>
          <ChevronDown className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} />
        </button>

        {/* Dropdown */}
        {open && (
          <div className="absolute z-50 mt-1 w-full min-w-[320px] sm:min-w-[380px] rounded-lg border border-border bg-popover text-popover-foreground shadow-lg animate-fade-in-up overflow-hidden">
            {/* Search bar */}
            <div className="p-2 border-b border-border">
              <div className="relative">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  ref={searchInputRef}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search voices or languages…"
                  className="h-8 pl-8 pr-8 text-sm"
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch("")}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            </div>

            {/* Filter bar: gender + group tabs */}
            <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border/50 overflow-x-auto scrollbar-none">
              {/* Gender filter pills */}
              <div className="flex items-center gap-1 flex-shrink-0">
                {(["all", "male", "female"] as const).map((g) => (
                  <button
                    key={g}
                    type="button"
                    onClick={() => setGenderFilter(g)}
                    className={cn(
                      "px-2 py-0.5 rounded-md text-[11px] font-medium transition-colors",
                      genderFilter === g
                        ? "bg-primary text-primary-foreground"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted"
                    )}
                  >
                    {g === "all" ? "All" : g === "male" ? "♂ Male" : "♀ Female"}
                  </button>
                ))}
              </div>

              <div className="w-px h-4 bg-border mx-1 flex-shrink-0" />

              {/* Group quick-jump pills */}
              <div className="flex items-center gap-0.5 overflow-x-auto scrollbar-none">
                <button
                  type="button"
                  onClick={() => setActiveGroup(null)}
                  className={cn(
                    "px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors whitespace-nowrap flex-shrink-0",
                    activeGroup === null
                      ? "bg-primary/20 text-primary"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  All langs
                </button>
                {VOICE_GROUPS.map((g) => (
                  <button
                    key={g.code}
                    type="button"
                    onClick={() => setActiveGroup(activeGroup === g.code ? null : g.code)}
                    className={cn(
                      "px-1.5 py-0.5 rounded text-[10px] font-medium transition-colors whitespace-nowrap flex-shrink-0",
                      activeGroup === g.code
                        ? "bg-primary/20 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {g.flag} {g.code.split("-")[0]}
                  </button>
                ))}
              </div>
            </div>

            {/* Results count */}
            <div className="px-3 py-1 text-[10px] text-muted-foreground/60">
              {totalFiltered} voice{totalFiltered !== 1 ? "s" : ""}
              {search && ` matching "${search}"`}
            </div>

            {/* Voice list */}
            <div
              ref={listRef}
              className="max-h-64 overflow-y-auto scrollbar-thin"
            >
              {filteredGroups.length === 0 ? (
                <div className="flex items-center justify-center py-8 text-sm text-muted-foreground">
                  No voices found
                </div>
              ) : (
                filteredGroups.map((group) => (
                  <div key={group.code}>
                    {/* Group header */}
                    <div className="sticky top-0 z-10 flex items-center gap-2 px-3 py-1.5 bg-muted/80 backdrop-blur-sm border-b border-border/30">
                      <span className="text-sm">{group.flag}</span>
                      <span className="text-xs font-semibold text-foreground">{group.label}</span>
                      <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                        {group.voices.length}
                      </Badge>
                    </div>

                    {/* Voice items in group */}
                    {group.voices.map((voice) => {
                      const isSelected = voice.value === value;
                      return (
                        <button
                          key={voice.value}
                          type="button"
                          onClick={() => handleSelect(voice.value)}
                          className={cn(
                            "w-full flex items-center gap-2 px-3 py-2 text-sm text-left transition-colors",
                            isSelected
                              ? "bg-primary/10 text-primary"
                              : "hover:bg-muted/50 hover:text-foreground"
                          )}
                        >
                          <div
                            className={cn(
                              "w-4 h-4 rounded-full border-2 flex items-center justify-center flex-shrink-0 transition-colors",
                              isSelected ? "border-primary bg-primary" : "border-muted-foreground/30"
                            )}
                          >
                            {isSelected && <Check className="h-2.5 w-2.5 text-primary-foreground" />}
                          </div>
                          <span className="flex-1 truncate">{voice.label}</span>
                          <span className={cn(
                            "text-[10px] px-1.5 py-0.5 rounded-full",
                            voice.gender === "male"
                              ? "bg-sky-500/10 text-sky-400"
                              : "bg-rose-500/10 text-rose-400"
                          )}>
                            {voice.gender === "male" ? "♂" : "♀"}
                          </span>
                          {voice.value.includes("Multilingual") && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                              ML
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      {/* Preview button */}
      <Button
        type="button"
        variant="outline"
        size="icon"
        onClick={onPreview}
        disabled={previewLoading}
        className="flex-shrink-0 h-9 w-9"
        title={previewLoading ? "Generating preview…" : previewPlaying ? "Stop preview" : "Preview voice"}
        aria-label={previewLoading ? "Generating voice preview" : previewPlaying ? "Stop voice preview" : "Play voice preview"}
      >
        {previewLoading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : previewPlaying ? (
          <Pause className="h-4 w-4" />
        ) : (
          <Volume2 className="h-4 w-4" />
        )}
      </Button>
    </div>
  );
}

// Re-export the flat voice list for the settings dialog and other consumers
export const VOICE_FLAT = VOICE_GROUPS.flatMap((g) =>
  g.voices.map((v) => ({
    value: v.value,
    label: `${g.flag} ${v.label} (${g.code}, ${v.gender})`,
  }))
);
