import time

def exec_time(start_time):
    # end time
    end_time = time.time()
    print()
    print("-" * 65)
    execution_time = end_time - start_time

    # convert time to HH:MM:SS.mmm format
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int((execution_time % 60) // 1)
    milliseconds = int((execution_time % 1) * 1000)

    # format to 2 digits (3 for milliseconds) with leading zeros
    formatted_hours = f"{hours:02}"
    formatted_minutes = f"{minutes:02}"
    formatted_seconds = f"{seconds:02}"
    formatted_milliseconds = f"{milliseconds:02}"

    # print execution time
    print("Execution time (HH:MM:SS.mmm): \033[92m{}:{}:{}.{}\033[0m".format(formatted_hours, formatted_minutes, formatted_seconds, formatted_milliseconds))
    print("-" * 65)
    print()