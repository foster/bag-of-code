Bag of Codes

### Explanation

The goal is to build a simple content-based recommendation engine which, 
given a public repository of javascript code, will recommend "similar" projects.

The notebook 01-prereqs will spider [npm](https://www.npmjs.com/)'s and record any
linked github repositories. Then it will clone those repositories and build a document term matrix from the source code.

### Setup

You should generate a [personal API token](https://github.com/blog/1509-personal-api-tokens) for github and add it as an environment variable. A lookup is performed against the github API to select for only javascript repositories (excluding coffeescript/typescript projects). The API token doesn't need any permissions, github just restricts anonymous use of their API.
Generate a token and save environment variables `GITHUB_API_USER` and `GITHUB_API_TOKEN`.
