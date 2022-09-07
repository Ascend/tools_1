import shlex
import subprocess


class CmdExecutor(object):
    @staticmethod
    def _change_cmd_format(cmds: str):
        cmds_list = []
        result = []

        if "|" in cmds:
            cmds_list = cmds.split("|")
        else:
            cmds_list.append(cmds)

        for cmd in cmds_list:
            cmd = shlex.split(cmd)
            result.append(cmd)

        return result

    @staticmethod
    def exec_cmd_get_out_put(cmds, wait):
        """
        通过cmd命令获取信息
        :param need_root_permission: 是否需要进行提权降权操作
        :param cmd: cmd命令
        :param wait: 设置cmd命令执行超时时间
        :return: 获得到的信息
        """
        ret = [0, "ok"]
        process_list = []
        try:
            cmds_tuple = tuple(CmdExecutor._change_cmd_format(cmds))
            for index, cmd in enumerate(cmds_tuple):
                if index == 0:
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                else:
                    p = subprocess.Popen(cmd, stdin=process_list[index - 1].stdout,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                process_list.append(p)

            if wait != 0:
                output, errout = process_list[-1].communicate(timeout=wait)
            else:
                output, errout = process_list[-1].communicate(timeout=300)
        except Exception as e:
            try:
                process_list[-1].kill()
            except Exception as ek:
                print("kill subprocess error")
            print("Call linux command error")
            return [-1000, "call linux command error"]

        if process_list[-1].returncode == 0:
            ret[1] = output.decode()
        else:
            ret[0] = process_list[-1].returncode
            ret[1] = output.decode() + errout.decode()
        return ret
