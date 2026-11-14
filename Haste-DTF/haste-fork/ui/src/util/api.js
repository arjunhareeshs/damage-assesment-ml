// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const APIUrl = import.meta.env.VITE_API_URL;
const APIMSubscriptionKey = import.meta.env.VITE_APIM_SUBSCRIPTION_KEY;

function resolveVarConcatChar(text) {
  if (text === "") return "";
  return text.includes("?") ? "&" : "?";
}

export function buildUrl(endpoint) {
  const base = APIUrl + endpoint;
  if (APIMSubscriptionKey) {
    const sep = base.includes("?") ? "&" : "?";
    return base + sep + "subscription-key=" + APIMSubscriptionKey;
  }
  return base;
}

// No sign-in: the app runs as a single local user with full access.
export const LOCAL_USER = {
  userId: "local-user",
  userRoles: ["administrators"],
  settings: { disableGuidedTour: false, itemsPerPage: 10 },
  status: "Active",
};

export async function apiValidateUser(setAppParams) {
  setAppParams((prevParams) => ({
    ...prevParams,
    userId: LOCAL_USER.userId,
    userRoles: LOCAL_USER.userRoles,
    userSettings: LOCAL_USER.settings,
    userStatus: LOCAL_USER.status,
  }));
}

export async function apiGet(endpoint) {
  try {
    const response = await fetch(buildUrl(endpoint));

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error fetching.:", error);
    throw new Error("Error fetching.");
  }
}

export async function apiPut(endpoint, data) {
  try {
    const response = await fetch(buildUrl(endpoint), {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    if (response.status === 409) {
      return response.status;
    }

    if (response.status !== 200 && response.status !== 409) {
      throw new Error(response);
    }
    const message = await response.json();
    return message;

  } catch (error) {
    throw new Error(error.message || "Error updating element.");
  }
}

export async function apiPost(endpoint, data, isFormData = false) {
  try {
    const options = {
      method: 'POST',
      headers: isFormData ? {} : { 'Content-Type': 'application/json' },
      body: isFormData ? data : JSON.stringify(data)
    };

    const response = await fetch(buildUrl(endpoint), options);
    if (!response.ok) {
      const message = await response.json();
      throw new Error(message.error || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error("Error uploading chunk.");
  }
}

export async function apiDelete(endpoint) {
  try {
    const response = await fetch(buildUrl(endpoint), {
      method: 'DELETE'
    });
    if (response.status !== 200) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response;
  } catch (error) {
    throw new Error("Error deleting element.");
  }
}