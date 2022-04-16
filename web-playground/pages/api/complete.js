import axios from "axios";
import getConfig from "next/config";

const { serverRuntimeConfig } = getConfig();

export default async function handler(req, res) {
  if (req.method == "POST") {
    const {
      body
    } = req;
    if (!body?.prompt) {
      res.statusCode = 400;
      res.end("Missing prompt");
      return;
    }
    const { apiUrl } = serverRuntimeConfig;
    try {
      const { data } = await axios.post(`${apiUrl}/complete`, body );
      res.status(200).json(data);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  } else {
    res.status(404);
  }
}
