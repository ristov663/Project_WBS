from django.db import models


class Institution(models.Model):
    name = models.CharField(max_length=255)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name


class Supplier(models.Model):
    name = models.CharField(max_length=255)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name


class Contract(models.Model):
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE)
    supplier = models.ForeignKey(Supplier, on_delete=models.CASCADE)
    date = models.DateField()
    amount = models.DecimalField(max_digits=15, decimal_places=2)
    description = models.TextField()

    def __str__(self):
        return f"{self.institution.name} - {self.supplier.name} - {self.date} ({self.amount} МКД)"
