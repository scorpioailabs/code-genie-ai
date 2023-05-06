# Code Genie AI

Code Genie AI is an automated tool that optimizes, refactors, and improves the readability of code. It solves boring refactoring jobs and helps developers work more efficiently.
We've forked McKay's repository repo-chatTo get started, check it out[link](https://github.com/mckaywrigley/repo-chat) to create this awesome forkage :D, full credit to him for the original work. We've added a few features to make it more useful for our use case.

## Requirements

In this project we use [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) and [Supabase with pgvector](https://supabase.com/docs/guides/database/extensions/pgvector) as our vector database.

You can switch out either of these with your own preference.

## How it works

Code Genie AI uses OpenAI embeddings to store codebases as vectors so that they can be retrieved and dependencies can be established. It automates code refactoring, offers suggestions, and provides unit tests (coming soon!).

## How to use

To use Code Genie AI, follow these steps:

1. Go to [Supabase](https://supabase.com/)
2. Create an account, if you don't already have one.
3. Once your account is created, click on **All projects>Create Project**.
4. Enter your project name, and Supabase will give you a URL and a service key.
5. Copy the `.env.example` file and rename it as `.env`.
6. Change the Supabase URL and the key in the `.env` file.
7. Configure the `.env` file with your repo url, repo branch of your choice, github pat_tokent (if you want to work on private repos), openai key, make sure you changed the Supabase's URL and key in the step 6.
8. Run `pip install -r requirements.txt` to install the dependencies

## About OpenAI API Key
Please ensure you are safely storing your api key and be careful with the the number of requests. As this project is in alpha, I highly recommend proceeding with caution when it comes to token use. Please refer to the OpenAI website on how to set limits.

## Contributing

Contributions to Code Genie AI are welcome! You can contribute by creating issues and submitting pull requests.

## License

Code Genie AI is licensed under the MIT License.

## Getting Help

If you need help using Code Genie AI, please create a GitHub issue and we will do our best to help you.
