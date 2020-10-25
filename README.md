<h1 align="center">SDSLabs-Issue-Labeler</h1>
<p align="center"><b>An ML Github bot to automate labelling of issues❤️</b></p>

### Overview

Maintaining proper labelled issues is difficult for large open source projects like Kubernetes, Docker, etc. This bot identifies the context of issue from its title and body and labels the issue accordingly. This ML bot uses Text-Multi-Classification-Algorithms to assign mutiple labels to a single issue. This bot responds to GitHub webhook events. When an issue is opened, ML bot predicts the appropriate labels and adds those labels to the issues.

### Techstack we used

- Google Bert Model (BERT-Base, UnCased ( 12-layer, 768-hidden, 12-heads , 110M parameters))
- GitHub API v3
- Python 3.6.9
- Tenserflow 1.15.2

### Dataset for model training

- Using GitHub API, we fetched issues of various repos along with the labels assigend to them by the maintainers and around 10,000 entries in a file.


### Supported labels

```enhancement```, ```bug```, ```feature```, ```good first issue```, ```question/discussion```, ```design```, ```help wanted```, ```high priority``` and ```documentation```

### Future scope

- Use Large Model of BERT.
- Training with larger and consistent dataset with cleaned up issue description.
- Add support to custom labels.

### :memo: License

Licensed under the [MIT License](./LICENSE).

Made with :heart: by SDSLabs.