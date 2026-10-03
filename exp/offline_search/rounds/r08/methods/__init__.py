"""R8 collection arms; simulator truth is restricted to OracleGraspCalls."""

from .methods import (AnchorCalls, EveryFiveAWM, FollowLottery, IdentificationProbe,
                      OracleGraspCalls, PolicyEveryTen, ShiftedAWM, WristEveryLook)

__all__ = ["AnchorCalls", "EveryFiveAWM", "FollowLottery", "IdentificationProbe",
           "OracleGraspCalls", "PolicyEveryTen", "ShiftedAWM", "WristEveryLook"]
