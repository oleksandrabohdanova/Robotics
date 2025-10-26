class Cost:
    def __init__(self, C, C_x, C_u, C_xx, C_ux, C_uu, Cf, Cf_x, Cf_xx):
        # Running cost and it's partial derivatives
        self.C = C
        self.C_prime = C_x, C_u, C_xx, C_ux, C_uu

        # Terminal cost and it's partial derivatives
        self.Cf = Cf
        self.Cf_prime = Cf_x, Cf_xx
