Bag of Codes

### Explanation

The goal is to build a simple content-based recommendation engine which, 
given a public repository of javascript code, will recommend "similar" projects.

It attempts [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
on a corpus of software packages. LSA is a technique for natural-language processing,
but here is applied corpus of documents written in a formal (artificial) language.

### Data Acquisition

[npm](https://www.npmjs.com/) is a popular package manager for node.js. A "package" is an
open source piece of software, usually intented to be used as component for other software.
The source code of a package on npm doesn't have to be hosted on [github](https://www.github.com/),
but they more than often (>90%) are.
By spidering npm's "most popular packages" web listing, we can accumulate a large list of github
urls to visit. Then, utilizing the github API, we can ensure a package:
1) is registered correctly (a small amount of packages list an invalid source url)
2) is actually written in javascript (see below)

Finally, we use the `git` command-line interface to clone all valid packages to a local location.
The complete source of every package is considered one "document" in terms of 

> A note on javascript.
> This project is only concerned with source code written in the Javascript programming language.
> This ensures both an apples-to-apples comparison between two repositories and ensures that
> we can correctly tokenize the source code. There are some very popular languages that transpile
> to javascript (coffeescript and typescript are two popular examples), and packages written in
> those languages are hosted on npm. Our data acquisition process intentionally skips packages
> written in those languages.

### Process

The notebook (bag-of-code.ipynb) will spider [npm] and record any linked github repositories.
Then it will clone those repositories and build a document term matrix from the source code.

### Setup

You should generate a [personal API token](https://github.com/blog/1509-personal-api-tokens) for github and add it as an environment variable. A lookup is performed against the github API to select for only javascript repositories (excluding coffeescript/typescript projects). The API token doesn't need any permissions, github just restricts anonymous use of their API.
Generate a token and save environment variables `GITHUB_API_USER` and `GITHUB_API_TOKEN`.
