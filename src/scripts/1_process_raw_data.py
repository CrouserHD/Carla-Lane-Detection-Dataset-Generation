import glob
import os
import sys
import numpy as np
import pygame
import carla
import tensorflow as tf
import cv2

# Verbindung zu CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Fahrzeug und Autopilot einrichten


def setup_vehicle():
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)
    return vehicle

# Kamera einrichten


def setup_camera(vehicle):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    spawn_point = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
    return camera

# Funktion zur Verarbeitung des Kamerabildes mit LaneNet


def process_image_with_lanenet(image, lanenet_model):
    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Preprocessing für LaneNet
    img_resized = cv2.resize(img_rgb, (512, 256)) / 127.5 - 1.0
    img_input = np.expand_dims(img_resized, axis=0)
    # Vorhersage mit LaneNet
    binary_seg_image = lanenet_model.predict(img_input)[0]
    binary_seg_image = (binary_seg_image > 0.5).astype(np.uint8) * 255
    binary_seg_image = cv2.resize(binary_seg_image, (image.width, image.height))
    # Überlagerung der erkannten Fahrspuren auf das Originalbild
    lanes_overlay = cv2.addWeighted(img_rgb, 1, cv2.cvtColor(binary_seg_image, cv2.COLOR_GRAY2BGR), 0.7, 0)
    return lanes_overlay

# Hauptlogik


def main():
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Lane Detection")
    clock = pygame.time.Clock()

    vehicle = setup_vehicle()
    camera = setup_camera(vehicle)

    # Laden des vortrainierten LaneNet-Modells
    lanenet_model = tf.keras.models.load_model('pfad_zum_vortrainierten_lanenet_modell.h5')

    def image_callback(image):
        processed_image = process_image_with_lanenet(image, lanenet_model)
        processed_image = np.rot90(processed_image)
        processed_image = pygame.surfarray.make_surface(processed_image)
        display.blit(processed_image, (0, 0))
        pygame.display.flip()

    camera.listen(image_callback)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            clock.tick(30)
    finally:
        camera.destroy()
        vehicle.destroy()
        pygame.quit()


if __name__ == '__main__':
    main()
