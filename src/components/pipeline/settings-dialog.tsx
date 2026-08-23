"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Settings, Loader2, Download, Upload, RotateCcw, X, FileText, Wifi, WifiOff, Zap } from "lucide-react";
import {
  Drawer,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerDescription,
  DrawerFooter,
  DrawerClose,
} from "@/components/ui/drawer";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";

const VOICES = [
  { value: "en-US-AndrewNeural", label: "en-US — Andrew" },
  { value: "en-US-JennyNeural", label: "en-US — Jenny" },
  { value: "en-US-GuyNeural", label: "en-US — Guy" },
  { value: "en-GB-RyanNeural", label: "en-GB — Ryan" },
  { value: "en-AU-WilliamNeural", label: "en-AU — William" },
  { value: "en-IN-NeerjaNeural", label: "en-IN — Neerja" },
  { value: "ja-JP-NanamiNeural", label: "ja-JP — Nanami" },
  { value: "ko-KR-SunHiNeural", label: "ko-KR — SunHi" },
  { value: "es-ES-ElviraNeural", label: "es-ES — Elvira" },
  { value: "zh-CN-XiaoxiaoNeural", label: "zh-CN — Xiaoxiao" },
];

const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
  { value: "de", label: "German" },
  { value: "pt", label: "Portuguese" },
  { value: "hi", label: "Hindi" },
  { value: "zh", label: "Chinese" },
];

interface SettingsForm {
  defaultVoice: string;
  defaultChapterLimit: number;
  defaultLanguage: string;
  groqKey: string;
  geminiKey: string;
  openRouterKey: string;
  zhipuKey: string;
  siliconFlowKey: string;
  autoArchive: boolean;
}

const DEFAULTS: SettingsForm = {
  defaultVoice: "en-US-AndrewNeural",
  defaultChapterLimit: 5,
  defaultLanguage: "en",
  groqKey: "",
  geminiKey: "",
  openRouterKey: "",
  zhipuKey: "",
  siliconFlowKey: "",
  autoArchive: false,
};

export function SettingsDialog() {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<SettingsForm>(DEFAULTS);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);
  const [ocrStatus, setOcrStatus] = useState<{ ready: boolean; model: string } | null>(null);
  const [ocrChecking, setOcrChecking] = useState(false);
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Listen for external open trigger (e.g. command palette)
  useEffect(() => {
    const handler = () => setOpen(true);
    window.addEventListener("open-settings-dialog", handler);
    return () => window.removeEventListener("open-settings-dialog", handler);
  }, []);

  const loadSettings = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/settings");
      if (res.ok) {
        const data = await res.json();
        const s = data.settings;
        setForm({
          defaultVoice: s.defaultVoice ?? DEFAULTS.defaultVoice,
          defaultChapterLimit: s.defaultChapterLimit ?? DEFAULTS.defaultChapterLimit,
          defaultLanguage: s.defaultLanguage ?? DEFAULTS.defaultLanguage,
          groqKey: s.groqKey ?? "",
          geminiKey: s.geminiKey ?? "",
          openRouterKey: s.openRouterKey ?? "",
          zhipuKey: s.zhipuKey ?? "",
          siliconFlowKey: s.siliconFlowKey ?? "",
          autoArchive: s.autoArchive ?? false,
        });
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  const checkOcrHealth = useCallback(async () => {
    setOcrChecking(true);
    try {
      const res = await fetch("/api/ocr/health?XTransformPort=3002");
      if (res.ok) {
        const data = await res.json();
        setOcrStatus({ ready: data.ready, model: data.model || "unknown" });
      } else {
        setOcrStatus({ ready: false, model: "unavailable" });
      }
    } catch {
      setOcrStatus({ ready: false, model: "unavailable" });
    } finally {
      setOcrChecking(false);
    }
  }, []);

  useEffect(() => {
    if (open) {
      loadSettings();
      checkOcrHealth();
    }
  }, [open, loadSettings, checkOcrHealth]);

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "Save failed");
      }
      toast({ title: "Settings saved", description: "Your preferences have been updated." });
      setOpen(false);
    } catch (e) {
      toast({
        title: "Failed to save settings",
        description: e instanceof Error ? e.message : "Unknown error",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleExport = useCallback(() => {
    const data = JSON.stringify(form, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "manhwa-recap-settings.json";
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Settings exported", description: "Your settings have been downloaded." });
  }, [form, toast]);

  const handleImport = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        setForm((prev) => ({
          ...prev,
          defaultVoice: data.defaultVoice ?? prev.defaultVoice,
          defaultChapterLimit: data.defaultChapterLimit ?? prev.defaultChapterLimit,
          defaultLanguage: data.defaultLanguage ?? prev.defaultLanguage,
          groqKey: data.groqKey ?? "",
          geminiKey: data.geminiKey ?? "",
          openRouterKey: data.openRouterKey ?? "",
          zhipuKey: data.zhipuKey ?? "",
          siliconFlowKey: data.siliconFlowKey ?? "",
          autoArchive: data.autoArchive ?? false,
        }));
        toast({ title: "Settings imported", description: "Review and save to apply." });
      } catch {
        toast({
          title: "Import failed",
          description: "Invalid settings file.",
          variant: "destructive",
        });
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  }, [toast]);

  const handleReset = useCallback(() => {
    setForm(DEFAULTS);
    toast({ title: "Settings reset", description: "Review and save to apply." });
  }, [toast]);

  const updateField = <K extends keyof SettingsForm>(key: K, value: SettingsForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
        aria-label="Open settings"
      >
        <Settings className="h-4 w-4" />
      </button>

      <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />

      <Drawer open={open} onOpenChange={setOpen} direction="right" dismissible handleOnly>
        <DrawerContent className="data-[vaul-drawer-direction=right]:sm:max-w-md overflow-hidden">
          <DrawerHeader className="flex-row items-center gap-2 px-6 py-4 shrink-0">
            <div className="p-1.5 rounded-lg bg-primary/10">
              <Settings className="h-4 w-4 text-primary" />
            </div>
            <div className="flex-1 min-w-0">
              <DrawerTitle>Settings</DrawerTitle>
              <DrawerDescription className="mt-0.5">
                Configure your default pipeline preferences and API keys.
              </DrawerDescription>
            </div>
            <DrawerClose asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8 shrink-0 rounded-lg hover:bg-muted/50">
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Button>
            </DrawerClose>
          </DrawerHeader>

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5 scrollbar-thin">
              {/* Voice & Language row */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="default-voice" className="text-xs font-medium">Default Voice</Label>
                  <Select
                    value={form.defaultVoice}
                    onValueChange={(v) => updateField("defaultVoice", v)}
                  >
                    <SelectTrigger id="default-voice" className="w-full">
                      <SelectValue placeholder="Select voice" />
                    </SelectTrigger>
                    <SelectContent className="max-h-48">
                      {VOICES.map((v) => (
                        <SelectItem key={v.value} value={v.value}>
                          {v.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="default-language" className="text-xs font-medium">Default Language</Label>
                  <Select
                    value={form.defaultLanguage}
                    onValueChange={(v) => updateField("defaultLanguage", v)}
                  >
                    <SelectTrigger id="default-language" className="w-full">
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent>
                      {LANGUAGES.map((l) => (
                        <SelectItem key={l.value} value={l.value}>
                          {l.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* OCR Engine Status */}
              <div className="rounded-xl border border-border bg-card/50 p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold flex items-center gap-2">
                    <Zap className="h-3.5 w-3.5 text-primary" />
                    OCR Transcription
                  </p>
                  <button
                    onClick={checkOcrHealth}
                    disabled={ocrChecking}
                    className="text-[10px] text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                  >
                    {ocrChecking ? "Checking…" : "Refresh"}
                  </button>
                </div>
                {ocrStatus ? (
                  <div className="flex items-center gap-2.5 text-xs">
                    {ocrStatus.ready ? (
                      <>
                        <Wifi className="h-3.5 w-3.5 text-emerald-400" />
                        <span className="text-emerald-400 font-medium">PaddleOCR {ocrStatus.model} Active</span>
                        <span className="text-muted-foreground/60">— primary transcriptor (no API keys needed)</span>
                      </>
                    ) : (
                      <>
                        <WifiOff className="h-3.5 w-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground">PaddleOCR not running — VLM providers will be used (API keys required)</span>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <FileText className="h-3.5 w-3.5" />
                    <span>PP-OCRv5 is the default transcription engine. VLM providers serve as fallback.</span>
                  </div>
                )}
              </div>

              <Separator />

              {/* Chapter limit */}
              <div className="space-y-2">
                <Label htmlFor="default-chapter-limit" className="text-xs font-medium">Default Chapter Limit</Label>
                <Input
                  id="default-chapter-limit"
                  type="number"
                  min={0}
                  value={form.defaultChapterLimit}
                  onChange={(e) =>
                    updateField(
                      "defaultChapterLimit",
                      Math.max(0, parseInt(e.target.value, 10) || 0)
                    )
                  }
                  className="max-w-xs"
                />
                <p className="text-[11px] text-muted-foreground/70">Set to 0 to process all chapters.</p>
              </div>

              <Separator />

              {/* API Keys */}
              <div className="space-y-4">
                <p className="text-sm font-semibold flex items-center gap-2">
                  <span className="text-primary">🔑</span> API Keys
                </p>

                <div className="space-y-2">
                  <Label htmlFor="groq-key" className="text-xs font-medium">Groq API Key</Label>
                  <Input
                    id="groq-key"
                    type="password"
                    placeholder="gsk_..."
                    value={form.groqKey}
                    onChange={(e) => updateField("groqKey", e.target.value)}
                  />
                  <p className="text-[11px] text-muted-foreground/70">
                    Faster VLM transcription. Free key at console.groq.com/keys
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="gemini-key" className="text-xs font-medium">Gemini API Key</Label>
                  <Input
                    id="gemini-key"
                    type="password"
                    placeholder="AI..."
                    value={form.geminiKey}
                    onChange={(e) => updateField("geminiKey", e.target.value)}
                  />
                  <p className="text-[11px] text-muted-foreground/70">
                    Second VLM provider for better transcription reliability.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="openrouter-key" className="text-xs font-medium">OpenRouter API Key</Label>
                  <Input
                    id="openrouter-key"
                    type="password"
                    placeholder="sk-or-..."
                    value={form.openRouterKey}
                    onChange={(e) => updateField("openRouterKey", e.target.value)}
                  />
                  <p className="text-[11px] text-muted-foreground/70">
                    For narration rewriting via language models.
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="zhipu-key" className="text-xs font-medium">Zhipu AI API Key</Label>
                  <Input
                    id="zhipu-key"
                    type="password"
                    placeholder="..."
                    value={form.zhipuKey}
                    onChange={(e) => updateField("zhipuKey", e.target.value)}
                  />
                  <p className="text-[11px] text-muted-foreground/70">
                    Free vision model (GLM-4V-Flash) for fast panel transcription.{' '}
                    <a href="https://open.bigmodel.cn" target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                      open.bigmodel.cn
                    </a>{' '}(free signup)
                  </p>
                </div>

                <div className="space-y-2 rounded-md border border-primary/20 bg-primary/5 p-2">
                  <Label htmlFor="siliconflow-key" className="text-xs font-medium flex items-center gap-1.5">
                    <span className="inline-flex items-center justify-center rounded bg-primary/20 text-[9px] font-bold text-primary px-1 py-0.5 leading-none">BEST FREE</span>
                    SiliconFlow API Key
                  </Label>
                  <Input
                    id="siliconflow-key"
                    type="password"
                    placeholder="sk-..."
                    value={form.siliconFlowKey}
                    onChange={(e) => updateField("siliconFlowKey", e.target.value)}
                  />
                  <p className="text-[11px] text-muted-foreground/70">
                    Free Qwen2.5-VL-7B (14M tokens/month). Primary VLM provider.{' '}
                    <a href="https://cloud.siliconflow.cn" target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                      cloud.siliconflow.cn
                    </a>
                  </p>
                </div>
              </div>

              <Separator />

              {/* Auto Archive toggle */}
              <div className="flex items-center justify-between gap-4 p-3 rounded-xl border border-border bg-card/50">
                <div className="space-y-0.5">
                  <Label htmlFor="auto-archive" className="text-xs font-medium">Auto Archive</Label>
                  <p className="text-[11px] text-muted-foreground/70">
                    Upload completed videos to cloud storage automatically.
                  </p>
                </div>
                <Switch
                  id="auto-archive"
                  checked={form.autoArchive}
                  onCheckedChange={(v) => updateField("autoArchive", v)}
                />
              </div>
            </div>
          )}

          {!loading && (
            <DrawerFooter className="flex-col sm:flex-row gap-2 px-6 border-t border-border">
              <div className="flex items-center gap-1 sm:ml-auto">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleReset}
                  className="h-8 text-xs"
                  title="Reset to defaults"
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  Reset
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleExport}
                  className="h-8 text-xs"
                  title="Export settings as JSON"
                >
                  <Download className="h-3 w-3 mr-1" />
                  Export
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="h-8 text-xs"
                  title="Import settings from JSON"
                >
                  <Upload className="h-3 w-3 mr-1" />
                  Import
                </Button>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setOpen(false)}>
                  Cancel
                </Button>
                <Button size="sm" onClick={handleSave} disabled={saving}>
                  {saving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  Save
                </Button>
              </div>
            </DrawerFooter>
          )}
        </DrawerContent>
      </Drawer>
    </>
  );
}