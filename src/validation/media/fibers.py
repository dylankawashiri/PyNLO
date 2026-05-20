from CuPyNLO.media.fibers.fiber import FiberInstance
from CuPyNLO.media.fibers.fiber_v2 import FiberInstance as FiberInstanceV2

fiber = FiberInstance()

fiber.load_from_db(10.0, "dudley")
print(fiber.get_gamma())
# for k, v in fiber.__dict__.items():
#     print(f"{k}:        {v}\n")

print("\nfiber_v2\n------------------\n")

fiber = FiberInstanceV2()
fiber.load_from_db(10.0, "dudley")

print(fiber.gamma)
# for k, v in fiber.__dict__.items():
#     print(f"{k}:        {v}\n")

