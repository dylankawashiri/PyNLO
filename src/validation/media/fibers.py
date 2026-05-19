from CuPyNLO.media.fibers.fiber import FiberInstance

fiber = FiberInstance()

fiber.load_from_db(10.0, "dudley")
print(fiber.get_gamma())
# for k, v in fiber.__dict__.items():
#     print(f"{k}:        {v}\n")

print("\nfiber_v2\n------------------\n")

from CuPyNLO.media.fibers.fiber_v2 import FiberInstance
fiber = FiberInstance()
fiber.load_from_db(10.0, "dudley")

print(fiber.gamma)
# for k, v in fiber.__dict__.items():
#     print(f"{k}:        {v}\n")

