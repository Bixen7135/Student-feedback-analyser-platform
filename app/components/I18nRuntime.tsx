"use client";

import { useEffect, useRef } from "react";
import { getDocumentTitle, translateText, type Locale } from "@/app/lib/i18n/catalog";
import { useI18n } from "@/app/lib/i18n/provider";

type TextState = {
  source: string;
  translated: string;
};

const textNodeState = new WeakMap<Text, TextState>();
const attributeState = new WeakMap<Element, Map<string, TextState>>();

const IGNORE_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT"]);
const ATTRIBUTE_NAMES = ["placeholder", "aria-label", "title"] as const;

function updateTextNode(node: Text, locale: Locale) {
  const currentValue = node.nodeValue ?? "";
  if (!currentValue.trim()) {
    return;
  }

  const previousState = textNodeState.get(node);
  const sourceValue =
    !previousState || (currentValue !== previousState.translated && currentValue !== previousState.source)
      ? currentValue
      : previousState.source;

  const translatedValue = translateText(sourceValue, locale);

  if (translatedValue !== currentValue) {
    node.nodeValue = translatedValue;
  }

  textNodeState.set(node, {
    source: sourceValue,
    translated: translatedValue,
  });
}

function updateAttribute(node: Element, attributeName: (typeof ATTRIBUTE_NAMES)[number], locale: Locale) {
  const currentValue = node.getAttribute(attributeName);
  if (!currentValue) {
    return;
  }

  const savedState = attributeState.get(node) ?? new Map<string, TextState>();
  const previousState = savedState.get(attributeName);
  const sourceValue =
    !previousState || (currentValue !== previousState.translated && currentValue !== previousState.source)
      ? currentValue
      : previousState.source;

  const translatedValue = translateText(sourceValue, locale);

  if (translatedValue !== currentValue) {
    node.setAttribute(attributeName, translatedValue);
  }

  savedState.set(attributeName, {
    source: sourceValue,
    translated: translatedValue,
  });

  attributeState.set(node, savedState);
}

function localizeNode(node: Node, locale: Locale) {
  if (node instanceof Text) {
    const parentElement = node.parentElement;
    if (!parentElement || parentElement.closest("[data-i18n-skip]")) {
      return;
    }
    if (IGNORE_TAGS.has(parentElement.tagName)) {
      return;
    }
    updateTextNode(node, locale);
    return;
  }

  if (!(node instanceof Element)) {
    return;
  }

  if (node.closest("[data-i18n-skip]")) {
    return;
  }

  if (IGNORE_TAGS.has(node.tagName)) {
    return;
  }

  for (const attributeName of ATTRIBUTE_NAMES) {
    updateAttribute(node, attributeName, locale);
  }

  if (node.tagName === "INPUT" || node.tagName === "TEXTAREA") {
    return;
  }

  for (const child of node.childNodes) {
    localizeNode(child, locale);
  }
}

export function I18nRuntime() {
  const { locale } = useI18n();
  const localeRef = useRef(locale);
  const frameRef = useRef<number | null>(null);
  const pendingNodesRef = useRef<Set<Node>>(new Set());

  useEffect(() => {
    localeRef.current = locale;
    document.title = getDocumentTitle(locale);
    localizeNode(document.body, locale);
  }, [locale]);

  useEffect(() => {
    const flush = () => {
      frameRef.current = null;
      const nodes = Array.from(pendingNodesRef.current);
      pendingNodesRef.current.clear();

      for (const node of nodes) {
        if (node instanceof Element && !node.isConnected) {
          continue;
        }
        localizeNode(node, localeRef.current);
      }
    };

    const schedule = (node: Node) => {
      pendingNodesRef.current.add(node);
      if (frameRef.current === null) {
        frameRef.current = window.requestAnimationFrame(flush);
      }
    };

    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === "characterData") {
          schedule(mutation.target);
          continue;
        }

        if (mutation.type === "attributes") {
          schedule(mutation.target);
          continue;
        }

        if (mutation.type === "childList") {
          mutation.addedNodes.forEach((node) => schedule(node));
        }
      }
    });

    observer.observe(document.body, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: [...ATTRIBUTE_NAMES],
    });

    return () => {
      observer.disconnect();
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  return null;
}
