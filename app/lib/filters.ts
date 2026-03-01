export type FilterOp = "eq" | "ne" | "contains" | "gt" | "lt" | "gte" | "lte";

export interface FilterRule {
  col: string;
  op: FilterOp;
  val: string;
}

export interface FilterStateSnapshot {
  filters: FilterRule[];
  search: string;
  sortCol: string;
  sortOrder: "asc" | "desc";
}

export function normalizeFilterRules(filters: FilterRule[]): FilterRule[] {
  return filters
    .map((rule) => ({
      col: String(rule.col ?? "").trim(),
      op: rule.op,
      val: String(rule.val ?? ""),
    }))
    .filter((rule) => rule.col.length > 0);
}

export function buildFilterSearchParams(
  state: Partial<FilterStateSnapshot>,
  seed?: URLSearchParams,
): URLSearchParams {
  const params = new URLSearchParams(seed);

  const filters = normalizeFilterRules(state.filters ?? []);
  if (filters.length > 0) {
    params.set("filters", JSON.stringify(filters));
  } else {
    params.delete("filters");
  }

  if (state.search && state.search.trim()) {
    params.set("search", state.search.trim());
  } else {
    params.delete("search");
  }

  if (state.sortCol && state.sortCol.trim()) {
    params.set("sort_col", state.sortCol.trim());
    params.set("sort_order", state.sortOrder === "desc" ? "desc" : "asc");
  } else {
    params.delete("sort_col");
    params.delete("sort_order");
  }

  return params;
}

export function parseFilterSearchParams(
  searchParams: URLSearchParams,
): FilterStateSnapshot {
  let filters: FilterRule[] = [];
  const rawFilters = searchParams.get("filters");
  if (rawFilters) {
    try {
      const parsed = JSON.parse(rawFilters);
      if (Array.isArray(parsed)) {
        filters = normalizeFilterRules(
          parsed.filter((item): item is FilterRule => {
            return Boolean(
              item &&
                typeof item === "object" &&
                "col" in item &&
                "op" in item &&
                "val" in item,
            );
          }),
        );
      }
    } catch {
      filters = [];
    }
  }

  const sortOrder = searchParams.get("sort_order") === "desc" ? "desc" : "asc";
  return {
    filters,
    search: searchParams.get("search") ?? "",
    sortCol: searchParams.get("sort_col") ?? "",
    sortOrder,
  };
}
