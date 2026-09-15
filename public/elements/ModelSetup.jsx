import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function ModelSetup() {
  const [provider, setProvider] = useState(props.provider || "google");
  const [model, setModel] = useState(props.model || "gemini-2.5-flash");
  const [apiKey, setApiKey] = useState("");

  const chooseProvider = (value) => {
    setProvider(value);
    setModel(value === "google" ? "gemini-2.5-flash" : "gpt-4.1-mini");
  };

  return (
    <div className="rounded-lg border p-4 space-y-4 max-w-xl">
      <div>
        <Label>Model provider</Label>
        <select
          className="mt-1 w-full rounded-md border bg-background p-2"
          value={provider}
          onChange={(event) => chooseProvider(event.target.value)}
        >
          <option value="google">Google Gemini</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>
      <div>
        <Label>Model name</Label>
        <Input
          value={model}
          onChange={(event) => setModel(event.target.value)}
          placeholder="gemini-2.5-flash or gpt-4.1-mini"
        />
      </div>
      <div>
        <Label>API key</Label>
        <Input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          placeholder="Stored only in this browser session"
          autoComplete="off"
        />
      </div>
      <Button
        disabled={!model.trim() || !apiKey.trim()}
        onClick={() => submitElement({ provider, model: model.trim(), api_key: apiKey.trim() })}
      >
        Start session
      </Button>
      <p className="text-xs text-muted-foreground">
        The key is kept in Chainlit session memory and is not written to project files.
      </p>
    </div>
  );
}

