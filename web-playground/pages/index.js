import React from "react";
import Head from "next/head";
import Checkbox from "@material-tailwind/react/Checkbox";
import Input from "@material-tailwind/react/Input";
import getConfig from "next/config";
import axios from "axios";

const { publicRuntimeConfig } = getConfig();

const THREAD_PREFIX = "-----";
const POST_PREFIX = "--- ";

function AutoGrowTextArea(props) {
  const textareaRef = React.useRef(null);

  React.useEffect(() => {
    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
  }, [props.value]);

  return <textarea ref={textareaRef} {...props}></textarea>;
}

function Post({ deletePost, postId, text, setText }) {
  return (
    <div className="flex flex-col border border-solid boder-b-2 border-stone-300 rounded my-5 p-5 pb-0">
      <h3 className="font-bold text-lg">{postId}</h3>
      <AutoGrowTextArea
        placeholder="Post text"
        className="border border-solid boder-b-2 w-full p-1"
        value={text}
        onChange={(ev) => setText(ev.target.value)}
      ></AutoGrowTextArea>
      <button
        type="button"
        onClick={deletePost}
        className="self-end border border-solid border-b-2 border-stone-500 rounded p-1 m-1"
      >
        Delete
      </button>
    </div>
  );
}

function RawForm({ rawText, setRawText }) {
  return (
    <div className="m-5">
      <AutoGrowTextArea
        placeholder="Prompt"
        className="border border-solid boder-b-2 w-full p-1"
        value={rawText}
        onChange={(ev) => setRawText(ev.target.value)}
      ></AutoGrowTextArea>
    </div>
  );
}

function ParsedForm({ posts, setPosts, setPostText, createPost }) {
  return (
    <div className="px-5">
      {posts.map((post) => (
        <Post
          key={post.postId}
          {...post}
          setText={(text) => setPostText(post.postId, text)}
          deletePost={() =>
            setPosts(posts.filter((p) => p.postId !== post.postId))
          }
        />
      ))}
      <button
        type="button"
        className="disabled:text-stone-300 border border-solid border-b-2 border-stone-500 rounded p-1"
        onClick={() => setPosts([...posts, createPost()])}
        disabled={!posts[posts.length - 1]?.text}
      >
        Add a post
      </button>
    </div>
  );
}

export default function Home() {
  const [showRaw, setShowRaw] = React.useState(false);
  const [loading, setLoading] = React.useState(false);

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

  const [rawText, setRawText] = React.useState();

  React.useEffect(() => {
    if (!rawText) {
      setRawText(THREAD_PREFIX);
      return;
    }
    if (!rawText.startsWith(THREAD_PREFIX)) {
      setRawText(THREAD_PREFIX + rawText);
      return;
    }
    const splits = rawText.split(THREAD_PREFIX);
    if (splits.length > 2) {
      // handle multi threads (discard all but the first)
      setRawText(THREAD_PREFIX + splits[1]);
      return;
    }
  }, [rawText]);

  async function generate(ev) {
    ev.preventDefault();
    setLoading(true);
    try {
      const {
        data: { response },
      } = await axios.post(`/api/complete`, {
        prompt: rawText,
        length: generationLength,
        top_p: generationP,
        temperature: generationTemperature,
      });
      if (!response) {
        return;
      }
      setRawText(rawText + response);
    } catch (e) {
      console.log(e);
    } finally {
      setLoading(false);
    }
  }

  function setPosts(posts) {
    const fullPrompt = getFullPrompt(posts);
    setRawText(fullPrompt);
  }

  function createPost() {
    return {
      postId: "" + Math.round(Math.random() * 1000000000),
      text: "",
    };
  }

  const threads = (rawText || "")
    .split(THREAD_PREFIX)
    .map((rawThread) => {
      if (rawThread.length === 0) {
        return null;
      }
      return rawThread
        .split("\n" + POST_PREFIX)
        .map((rawPost) => {
          if (rawPost.length === 0) {
            return null;
          }
          const post = createPost();
          const match = rawPost.match(/^\s?(\d+)\n([\s\S]*)/);
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
    const postIndex = posts.findIndex((p) => p.postId === postId);
    const post = posts[postIndex];
    const newPost = { ...post, text };
    setPosts([
      ...posts.slice(0, postIndex),
      newPost,
      ...posts.slice(postIndex + 1),
    ]);
  }

  function getFullPrompt(posts) {
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
        <div className="flex flex-col text-center mt-5">
          <h1 className="text-3xl font-bold">GPT-4chan Playground</h1>
          <p className="">This is a playground for GPT-4chan.</p>
        </div>
        <div className="flex justify-end mx-5">
          <Checkbox
            color="gray"
            text="Show raw text"
            id="show-raw"
            checked={showRaw}
            onChange={(ev) => setShowRaw(ev.target.checked)}
          />
        </div>
        {showRaw ? (
          <RawForm rawText={rawText} setRawText={setRawText} />
        ) : (
          <ParsedForm
            posts={posts}
            setPosts={setPosts}
            setPostText={setPostText}
            createPost={createPost}
          />
        )}
        <div className="flex flex-col md:flex-row m-5">
          {generationSettings.map(({ label, value, setValue, min, max }) => (
            <div key={label} className="w-full m-1 p-1">
              <Input
                type="number"
                placeholder={label}
                value={value}
                onChange={(ev) => setValue(ev.target.value)}
                min={min}
                max={max}
              />
            </div>
          ))}
        </div>
        <div className="flex m-5">
          <button
            type="button"
            onClick={generate}
            className="border border-solid boder-b-2 border-stone-300 rounded px-5 py-2 disabled:text-stone-300"
            disabled={loading}
          >
            Generate
          </button>
        </div>
        {publicRuntimeConfig.showDebugPrompt && (
          <div className="whitespace-pre-wrap">{rawText}</div>
        )}
      </main>

      <footer className=""></footer>
    </div>
  );
}
