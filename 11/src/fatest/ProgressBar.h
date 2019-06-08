#ifndef PROGRESS_BAR_H_
#define PROGRESS_BAR_H_

class ProgressBarPrinter {
public:
    ProgressBarPrinter(int total_items, int images_per_item) :
        total_items(total_items), images_per_item(images_per_item), start_time(std::chrono::high_resolution_clock::now()) {}
    
    void Print(int progress)
    {
        if (progress == 0)
        {
            std::cout << "Progress: 0% | 0/" << total_items << "\r" << std::flush;
        }
        else
        {
            int items_finished = int(progress / (images_per_item + 1));
            int percentage_finished = int(items_finished / float(total_items) * 100);

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start_time;
            double passed_time = elapsed.count();
            
            double time_per_item = passed_time / items_finished;
            times_per_item.push_back(time_per_item);
            if (times_per_item.size() > 20) times_per_item.erase(times_per_item.begin());
            time_per_item = std::accumulate(times_per_item.begin(), times_per_item.end(), 0.0) / times_per_item.size();
            
            double time_remaining = time_per_item * (total_items - items_finished);
            int minutes_remaining = int(time_remaining / 60);
            int seconds_remaining = int(time_remaining) % 60;

            std::cout
                << "Progress: "
                << percentage_finished << "% | " << items_finished << "/" << total_items
                << " | Remaining time: " << minutes_remaining << ":" << (seconds_remaining < 10 ? "0" : "") << seconds_remaining
                << " | Time per item: " << time_per_item
                << " | Time per image: " << time_per_item / double(images_per_item)
                << "\r" << std::flush;
        }
    }

    void RestartTime()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

private:
    using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;
    Time start_time;
    int total_items;
    std::vector<double> times_per_item;
    int images_per_item;
};

#endif /* PROGRESS_BAR_H_ */
