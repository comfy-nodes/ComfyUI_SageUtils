This is the Sage Utils set of custom nodes. Everyone seems to be making their own suites of nodes, and I decided I'd go ahead and make one myself, for any nodes I feel would be helpful.

I've started with a bunch of metadata nodes. There are several other node sets out there for adding metadata for interoperability with Civitai, but I wasn't happy with them and decided to try spinning my own, though I definitely looked at ImageSaver and other places.

Anything dealing with Lora Stacks is intended to be interoperable with other custom nodes that use those, though I'm also spinning my own versions of a few nodes so that you don't need to use other sets nodes with mine.

There's an example workflow in the example folder, and I talk about how to use them below.

![Example Workflow Image](examples/example_workflow.png)

The current workflow for usage of these looks like this:

Use the "Load Checkpoint with Name" node to load your model. It's got a special "model_info" output that has the path to the model and the model's sha256 hash. This is a custom node because I was unable to find any way to get the model path from the model output.

For Loras, right now, you can either string together the "Simple Lora Stack" node I provided, or use ones from other custom node packs. These are definitely compatable with [ComfyUI-Lora-Auto-Trigger-Words](https://github.com/idrirap/ComfyUI-Lora-Auto-Trigger-Words), and undoubtedly others.

The lora_stack output at the end of the chain can be connected to my "Lora Stack Loader" node, along with the model and clip, or you could use a similar node from another node pack, like the one in [Comfyroll Studio](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes).

With the positive and negative prompt, you'll want to put them in a node that let's you put text in it, such as my "Set Text" node, or one of the millions of others out there. The one in ComfyUI-Lora-Auto_Trigger_Words is nice. You could connect those to two normal Clip Text Encode nodes, or use my "Prompts to Clip" node, which is less space, and passes through the text you give it for easy routing.

We're also using a custom KSampler node. I've split it in two: a "Sampler Info" node, and a "KSampler w/ Sampler Info" node. You *could* use a normal KSampler, but if you do, you'll still need to set all the settings on that "Sampler Info" node, as we need the output. I'm thinking about adding alternatives here.

Now, we're going to hook that "model_info" output, the "lora_stack" node (if you have loras), the prompt text outputs, and the width and height up to the "Construct Metadata" node, and that outputs "param_metadata", which is approximating what the metadata from A1111 would look like, including models and loras with hashes, so Civitai will pick them up as resources if you post a picture there. Note that that output is just a string. You can use any nodes to modify strings on it.

And finally, we run everything to the "Save Image w/ Added Metadata" node. This acts just like a normal Save Images node, except that it has a "param_metadata" input and and "extra_metadata" input. Both are regular strings, with the first saving to the "parameters" metadata keyword, and the second to "extra". In theory, you could hook anything to them, but the first is obviously intended for the output of the "Construct Metadata Node".

There's also a utility node to calculate the sha256 of a file seperately, which could be used if you wanted to create the metadata yourself. I could potentially look up the model on civitai with that in the future. I'll undoubtedly be adding more nodes here and there.

Feel free to file issues if you have idea or run into bugs, or, better yet, pr's, though there's no guarantee I'll accept them.

I'm also currently unemployed, so feel free to toss some money my way to my [Kofi](https://ko-fi.com/arcum42) account, through other ways, or, indeed, job offers.