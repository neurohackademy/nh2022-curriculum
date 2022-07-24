[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/carpentries-incubator/docker-introduction)

# Reproducible computational environments using containers: Introduction to Docker

This directory contains the Docker tutorial for the 2021 NeuroHackademy. The tutorial is borrowed from [the Carpentries](https://carpentries.org/) and has been lightly adapted and extended by [Noah C. Benson](https://github.com/noahbenson/). The currently-maintained official version of this tutorial can be found at the [Carpentries Incubator GitHub page](https://github.com/carpentries-incubator/docker-introduction). A copy of this tutorial as-is may also be found [here on GitHub](https://github.com/richford/docker-introduction).

This directory itself is a Jekyll website; however it can be run as a web-server using Docker quite easily; if you have Docker installed and running, you can do the following to start the web-server:

```bash
# In BASH:
$ cd nh2021-curriculum/docker-tutorial
$ docker compose up
```

These commands will produce a large amount of output, but eventually you will be able to point your browser to [localhost:4000](http://127.0.0.1:4000/) to see the lesson content. This content can also be found [here](https://richiehalford.org/docker-introduction/).

---

[![Create a Slack Account with us](https://img.shields.io/badge/Create_Slack_Account-The_Carpentries-071159.svg)](https://swc-slack-invite.herokuapp.com/)

This repository generates the corresponding lesson website from [The Carpentries](https://carpentries.org/) repertoire of lessons. 

If you are interested in Singularity as opposed to Docker, see the Singularity lesson in the Carpentries Incubator: 
* [Reproducible Computational Environments Using Containers: Introduction to Singularity](https://github.com/carpentries-incubator/singularity-introduction)

## Contributing

We welcome all contributions to improve the lesson! Maintainers will do their best to help you if you have any
questions, concerns, or experience any difficulties along the way.

We'd like to ask you to familiarize yourself with our [Contribution Guide](CONTRIBUTING.md) and have a look at
the [more detailed guidelines][lesson-example] on proper formatting, ways to render the lesson locally, and even
how to write new episodes.

Please see the current list of [issues](https://github.com/carpentries-incubator/docker-introduction/issues) for ideas for contributing to this
repository. For making your contribution, we use the GitHub flow, which is
nicely explained in the chapter [Contributing to a Project](http://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project) in Pro Git
by Scott Chacon.
Look for the tag ![good_first_issue](https://img.shields.io/badge/-good%20first%20issue-gold.svg). This indicates that the mantainers will welcome a pull request fixing this issue.  


## Maintainer(s)

Current maintainers of this lesson are 

* [David Eyers](https://github.com/dme26/)
* [Sarah Stevens](https://github.com/sstevens2/)


## Authors

A list of contributors to the lesson can be found in [AUTHORS](AUTHORS)

## Citation

To cite this lesson, please consult with [CITATION](CITATION)

[lesson-example]: https://carpentries.github.io/lesson-example
