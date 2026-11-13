"use client";

import { useEffect } from "react";
import { mountWidgets } from "@/lib/widget-runtime";

// Builds the interactive widgets of the current page.
export function Widgets({ names }: { names: string[] }) {
  useEffect(() => {
    mountWidgets(names);
  }, [names]);
  return null;
}
