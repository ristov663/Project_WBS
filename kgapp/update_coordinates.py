import time
import requests
import os
import sys
import django

# Set up Django environment
sys.path.append(os.path.abspath('C:/Users/pc/Desktop/WbsProject'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WbsProject.settings')
django.setup()

from kgapp.models import Institution, Supplier


def get_coordinates(name):
    """Function to get coordinates using OpenStreetMap Nominatim API"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': name,
        'format': 'json',
        'addressdetails': 1,
        'limit': 1,
    }
    headers = {
        'User-Agent': 'MyApp ristovbojan663@gmail.com'
    }
    try:
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return float(data['lat']), float(data['lon'])

    except Exception as e:
        print(f"Error getting coordinates for {name}: {e}")
    return None, None


def update_coordinates():
    """Function to update coordinates in the database"""

    # Update institutions
    institutions = Institution.objects.filter(latitude__isnull=True, longitude__isnull=True)
    for institution in institutions:
        lat, lon = get_coordinates(institution.name)
        if lat and lon:
            institution.latitude = lat
            institution.longitude = lon
            institution.save()

    # Update suppliers
    suppliers = Supplier.objects.filter(latitude__isnull=True, longitude__isnull=True)
    for supplier in suppliers:
        lat, lon = get_coordinates(supplier.name)
        if lat and lon:
            supplier.latitude = lat
            supplier.longitude = lon
            supplier.save()
            print(f"Updated {supplier.name} with coordinates: ({lat}, {lon})")
        time.sleep(1)  # Sleep to avoid overloading the API


if __name__ == "__main__":
    update_coordinates()
