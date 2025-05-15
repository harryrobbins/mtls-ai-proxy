I have this project in which I'm trying to build a custom handler for LiteLLM that will do two things:



 - provide mtls authentication to my custom openapi endpoint
 - patch the streaming method so that it returns what looks like a stream, but, under the hood, it will acutally just call an async completion from my upstream llm provider and return it as a single chunk (my upstream provider doesn't support streaming but i want that to be transparent to my client app)
