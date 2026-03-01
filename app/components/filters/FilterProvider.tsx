"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  usePathname,
  useRouter,
  useSearchParams,
} from "next/navigation";

import {
  buildFilterSearchParams,
  FilterRule,
  parseFilterSearchParams,
} from "@/app/lib/filters";

interface FilterContextValue {
  filters: FilterRule[];
  search: string;
  sortCol: string;
  sortOrder: "asc" | "desc";
  setFilters: React.Dispatch<React.SetStateAction<FilterRule[]>>;
  setSearch: React.Dispatch<React.SetStateAction<string>>;
  setSortCol: React.Dispatch<React.SetStateAction<string>>;
  setSortOrder: React.Dispatch<React.SetStateAction<"asc" | "desc">>;
  replaceAll: (next: {
    filters?: FilterRule[];
    search?: string;
    sortCol?: string;
    sortOrder?: "asc" | "desc";
  }) => void;
  clearAll: () => void;
}

const FilterContext = createContext<FilterContextValue | null>(null);

export function FilterProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const initialState = useMemo(
    () => parseFilterSearchParams(new URLSearchParams(searchParams.toString())),
    [searchParams],
  );

  const [filters, setFilters] = useState<FilterRule[]>(initialState.filters);
  const [search, setSearch] = useState(initialState.search);
  const [sortCol, setSortCol] = useState(initialState.sortCol);
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">(initialState.sortOrder);
  const lastQueryRef = useRef<string>(buildFilterSearchParams(initialState).toString());

  useEffect(() => {
    const nextParams = buildFilterSearchParams({
      filters,
      search,
      sortCol,
      sortOrder,
    });
    const nextQuery = nextParams.toString();
    if (nextQuery === lastQueryRef.current) {
      return;
    }
    lastQueryRef.current = nextQuery;
    const nextUrl = nextQuery ? `${pathname}?${nextQuery}` : pathname;
    router.replace(nextUrl, { scroll: false });
  }, [filters, pathname, router, search, sortCol, sortOrder]);

  const value = useMemo<FilterContextValue>(
    () => ({
      filters,
      search,
      sortCol,
      sortOrder,
      setFilters,
      setSearch,
      setSortCol,
      setSortOrder,
      replaceAll(next) {
        setFilters(next.filters ?? []);
        setSearch(next.search ?? "");
        setSortCol(next.sortCol ?? "");
        setSortOrder(next.sortOrder ?? "asc");
      },
      clearAll() {
        setFilters([]);
        setSearch("");
        setSortCol("");
        setSortOrder("asc");
      },
    }),
    [filters, search, sortCol, sortOrder],
  );

  return (
    <FilterContext.Provider value={value}>{children}</FilterContext.Provider>
  );
}

export function useFilterContext(): FilterContextValue {
  const context = useContext(FilterContext);
  if (!context) {
    throw new Error("useFilterContext must be used inside FilterProvider.");
  }
  return context;
}
