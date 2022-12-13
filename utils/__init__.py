import logging
import typing
import abc
import pyjapc

__version__ = "0.0.5.dev0"


class Device(abc.ABC):
    def __init__(
        self,
        japc: typing.Optional[pyjapc.PyJapc],
        name: str,
        timingSelectorOverride: str = None,
    ):
        """Generic Japc device interface

        Args:
            japc (typing.Optional[pyjapc.PyJapc]): Instance of PyJapc object from main
            name (str): Name of the device
            timingSelectorOverride (str, optional): String to override the timing selector. Defaults to None.
        """
        self.japc = japc
        if timingSelectorOverride:
            self.cycle_selector_override = timingSelectorOverride
        else:
            self.cycle_selector_override = self.japc.getSelector()
        logging.debug(self.cycle_selector_override)
        self.name = name

    def print_name(self) -> None:
        """Function to print name"""
        print(self.name)

    def __str__(self) -> str:
        return self.name

    def setParameter(self, **kwargs) -> None:
        """Wrapper around the setParam of PyJapc"""
        pass

    @abc.abstractmethod
    def getParameter(
        self, ele: str = None, getHeader: bool = True, **kwargs
    ) -> typing.Tuple[any, dict]:
        """Wrapper arounf the getParam method of PyJapc

        Args:
            ele (str, optional): String representing the device field (e.g. device/property#field). If None, then default action is performed. Defaults to None.
            getHeader (bool, optional): Boolean to get header from device field. Defaults to True.

        Returns:
            typing.Tuple[any, dict]: Tuple like (data, info).
        """
        pass
