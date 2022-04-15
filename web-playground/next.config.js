/** @type {import('next').NextConfig} */

const toBoolean = require('to-boolean');

const nextConfig = {
  reactStrictMode: true,
  serverRuntimeConfig: {
    apiUrl: process.env.API_URL, // Pass through env variables
  },
  publicRuntimeConfig: {
    showDebugPrompt: toBoolean(process.env.SHOW_DEBUG_PROMPT || "false"),
  },
}

module.exports = nextConfig
