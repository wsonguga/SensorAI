class DotLicense:
    def __init__(self, authorization_file="", tpm_rsa_ctx_file=""):
        self.tpm_public_key=authorization_file
        self.aa=tpm_rsa_ctx_file
        

    def runtime_verify(self):
        return True

    
    def number_of_devices(self):
        return 999999


    def get_expiration_date(self):
       return 0