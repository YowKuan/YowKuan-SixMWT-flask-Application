from app import init_app


web = init_app()


if __name__ == "__main__":
    web.run(host='0.0.0.0', port = 8000)