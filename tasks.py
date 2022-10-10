from invoke import task
import os


def parse_env(env_check=True):
    circle_branch = os.environ.get("CIRCLE_BRANCH")
    env_stage = "prd" if circle_branch == _PRD_GIT_BRANCH else circle_branch

    if env_stage not in ["dev", "stg", "prd"]:
        if env_check:
            raise ValueError("Only dev, stg and prd (main) branches can be deployed.")

    return env_stage


@task
def format(c, check=False):
    if check:
        c.run("black --check src/ tasks.py")
    else:
        c.run("black src/ tasks.py")


@task
def lint(c):
    c.run("flake8 src/ tasks.py")


@task
def wheel(c):
    c.run("python setup.py bdist_wheel")


@task
def clean(c):
    c.run("rm -r dist/ build/ *.egg-info/")
