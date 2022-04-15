import React from "react";
import Head from "next/head";
import Checkbox from "@material-tailwind/react/Checkbox";
import Input from "@material-tailwind/react/Input";
import getConfig from 'next/config';
import axios from "axios";

const { publicRuntimeConfig } = getConfig()

const THREAD_PREFIX = "-----";
const POST_PREFIX = "--- ";

function Post({ deletePost, postId, text, setText }) {
  return (
    <div className="">
      <div>{POST_PREFIX + postId}</div>
      <textarea
        placeholder="Type here..."
        className="border border-solid boder-b-2"
        value={text}
        onChange={(ev) => setText(ev.target.value)}
      ></textarea>
      <button type="button" onClick={deletePost}>
        Delete
      </button>
    </div>
  );
}

function RawForm({ rawText, setRawText }) {
  return (
    <div className="">
      <textarea
        placeholder="Type here..."
        className="border border-solid boder-b-2"
        value={rawText}
        onChange={(ev) => setRawText(ev.target.value)}
      ></textarea>
    </div>
  );
}

function ParsedForm({ posts, setPosts, setPostText }) {
  return (
    <div className="">
      <div>-----</div>
      {posts.map((post) => (
        <Post
          key={post.postId}
          {...post}
          setText={(text) => setPostText(post.postId, text)}
          deletePost={() =>
            setPosts((posts) => posts.filter((p) => p.postId !== post.postId))
          }
        />
      ))}
      <button
        type="button"
        className="disabled:color-stone-300"
        onClick={() => setPosts((posts) => [...posts, createPost()])}
        disabled={!posts[posts.length - 1]?.text}
      >
        Add
      </button>
    </div>
  );
}

export default function Home() {

  const [showRaw, setShowRaw] = React.useState(false);


  const [generationLength, setGenerationLength] = React.useState(128);
  const [generationP, setGenerationP] = React.useState(0.8);
  const [generationTemperature, setGenerationTemperature] = React.useState(2.0);
  const generationSettings = [
    {
      label: "Length",
      value: generationLength,
      setValue: setGenerationLength,
      min: 1,
      max: 1024,
    },
    {
      label: "topP",
      value: generationP,
      setValue: setGenerationP,
      min: 0.0,
      max: 1.0,
    },
    {
      label: "Temperature",
      value: generationTemperature,
      setValue: setGenerationTemperature,
      min: 0.0,
    },
  ];

  const [rawText, setRawText] = React.useState(
    "-----\n--- 384284343\nHello\nwhy"
  );

  async function handleSubmit(ev) {
    ev.preventDefault();
    const { data: {response} } = await axios.post(
      `/api/complete`,
      {
        prompt: rawText,
        length: generationLength,
        top_p: generationP,
        temperature: generationTemperature,
      }
    );
    if(!response) {
      return;
    }
    setRawText(rawText + response);
  }

  function setPosts() {}

  function createPost() {
    return {
      postId: "" + Math.round(Math.random() * 1000000000),
      text: "",
    };
  }

  const threads = rawText
    .split(THREAD_PREFIX + "\n")
    .map((rawThread) => {
      if (rawThread.length === 0) {
        return null;
      }
      return rawThread
        .split(POST_PREFIX)
        .map((rawPost) => {
          if (rawPost.length === 0) {
            return null;
          }
          const post = createPost();
          const match = rawPost.match(/^(\d+)\n([\s\S]*)/);
          if (match) {
            post.postId = match[1];
            post.text = match[2];
          }
          return post;
        })
        .filter((post) => post !== null);
    })
    .filter((thread) => thread !== null);

  const posts = threads.length ? threads[0] : [];

  function setPostText(postId, text) {
    setPosts((posts) => {
      const postIndex = posts.findIndex((p) => p.postId === postId);
      const post = posts[postIndex];
      const newPost = { ...post, text };
      return [
        ...posts.slice(0, postIndex),
        newPost,
        ...posts.slice(postIndex + 1),
      ];
    });
  }

  function getFullPrompt() {
    return [
      THREAD_PREFIX,
      ...posts.map((post) => `${POST_PREFIX} ${post.postId}\n${post.text}`),
    ].join("\n");
  }

  React.useEffect(() => {
    if (posts.length === 0) {
      setPosts([createPost()]);
    }
  }, [posts]);

  return (
    <div className="">
      <Head>
        <title>GPT-4chan Playground</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="">
        <div className="">
          <h1 className="">GPT-4chan Playground</h1>
          <p className="">This is a playground for GPT-4chan.</p>
        </div>
        <Checkbox
          color="gray"
          text="Show raw text"
          id="show-raw"
          checked={showRaw}
          onChange={(ev) => setShowRaw(ev.target.checked)}
        />
        {showRaw ? (
          <RawForm rawText={rawText} setRawText={setRawText} />
        ) : (
          <ParsedForm
            posts={posts}
            setPosts={setPosts}
            setPostText={setPostText}
            handleSubmit={handleSubmit}
          />
        )}
        <div>
          {generationSettings.map(({label, value, setValue, min, max}) => (
            <div key={label}>
            <Input type="number" placeholder={label} value={value} onChange={(ev) => setValue(ev.target.value)} min={min} max={max}/>
            </div>
          ))}
        </div>
        <button type="button" onClick={handleSubmit} className="border border-solid boder-b-2 border-stone-300 rounded px-5 py-2">
          Generate
        </button>
        {publicRuntimeConfig.showDebugPrompt && <div className="whitespace-pre-wrap">{getFullPrompt()}</div>}
      </main>

      <footer className=""></footer>
    </div>
  );
}
