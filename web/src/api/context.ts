/**
 * Which API client the components use.
 *
 * The app uses the real client (`api`, which calls /api). Tests put a client
 * with a fake `fetch` into this context instead, so they exercise the real
 * request and error handling and only replace the network.
 */
import { createContext, useContext } from "react";

import { api, type ApiClient } from "./client";

export const ApiContext = createContext<ApiClient>(api);

export function useApi(): ApiClient {
  // useContext: reads the client from the nearest <ApiContext> above, so a
  // component doesn't import a global it can't swap out in tests.
  return useContext(ApiContext);
}
