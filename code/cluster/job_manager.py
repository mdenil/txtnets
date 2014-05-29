__author__ = 'mdenil'

import os
import stat
import yaml
import subprocess
import jinja2 as j


class Task(object):
    def __init__(self,
            job_dir,
            templates):
        self.job_dir = job_dir
        self.templates = templates

    def configure(self, params=None):
        for template in self.templates:
            template.configure(params)

    def launch(self):
        raise NotImplementedError("Not Implemented")

    def finished(self):
        raise NotImplementedError("Not Implemented")

    def result(self):
        raise NotImplementedError("Not Implemented")


class LocalTask(Task):
    def __init__(self, launcher_file, *args, **kwargs):
        super(LocalTask, self).__init__(*args, **kwargs)
        self.launcher_file = launcher_file
        self.launcher_path = os.path.join(self.job_dir, launcher_file)

    def launch(self):
        st = os.stat(self.launcher_path)
        os.chmod(self.launcher_path, st.st_mode | stat.S_IXUSR)

        subprocess.check_call(self.launcher_path)


class ClusterTask(Task):
    def __init__(self, launcher_file, *args, **kwargs):
        super(ClusterTask, self).__init__(*args, **kwargs)
        self.launcher_file = launcher_file
        self.launcher_path = os.path.join(self.job_dir, launcher_file)

    def launch(self):
        st = os.stat(self.launcher_path)
        os.chmod(self.launcher_path, st.st_mode | stat.S_IXUSR)

        subprocess.check_call(["qsub", self.launcher_path])


class Template(object):
    def __init__(self, env, target, src, params, params_target=None):
        self.env = env
        self.target = target
        self.src = src
        self.params = params
        self.params_target = params_target

        self._template = self.env.get_template(self.src)

    def _prepare_params(self, params):
        params = dict() if params is None else params
        return _merge_params(self.params, params)

    def _configure(self, params):
        return self._template.render(**self._prepare_params(params))

    def configure(self, params):
        with open(self.target, 'wb') as target:
            target.write(self._configure(params))

        if self.params_target is not None:
            with open(self.params_target, 'wb') as params_target:
                yaml.dump(
                    self._prepare_params(params),
                    params_target)


class Job(object):
    def __init__(
            self,
            job_id,
            base_dir,
            params,
            template_dir,
            tasks,
            task_factory):

        self.job_id = "{:08}".format(job_id)
        self.base_dir = base_dir
        self._tasks = iter(tasks)
        self.task_factory = task_factory
        self.job_dir = os.path.abspath(os.path.join(self.base_dir, self.job_id))
        self.global_params = params.copy()

        assert "job_id" not in self.global_params
        self.global_params["job_id"] = self.job_id

        assert "job_dir" not in self.global_params
        self.global_params["job_dir"] = self.job_dir

        self.env = j.Environment(
            loader=j.FileSystemLoader(template_dir),
            undefined=j.StrictUndefined)
        self.env.globals.update(self.global_params)

        # ensure our working space exists
        try:
            os.makedirs(self.job_dir)
        except OSError:
            # if the dir already exists makedirs throws an error, carry on
            pass

    def tasks(self):
        for task in self._tasks:
            templates = []
            for info in task['templates']:
                template = Template(
                    env=self.env,
                    target=os.path.join(self.job_dir, info['target']),
                    params_target=os.path.join(self.job_dir, info['params_target']),
                    src=info['src'],
                    params=_merge_params(self.global_params, info['params']))
                templates.append(template)

            yield self.task_factory(
                    job_dir=self.job_dir,
                    templates=templates,
                    **task['task_params'])


################################
# Useful functions

def _merge_params(*params):
    all_params = dict()
    for package in params:
        for k,v in package.iteritems():
            all_params[k] = v
    return all_params


